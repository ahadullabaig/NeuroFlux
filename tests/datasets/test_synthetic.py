"""Comprehensive tests for synthetic signal generators (Phase 2)."""

import numpy as np
import pytest
from neuroflux_core.metrics.snr import calculate_snr
from neuroflux_datasets.synthetic import (
    InterferenceKind,
    SignalGenerator,
    add_flicker,
    add_gaussian_noise,
    add_interference,
    generate_ecg,
    generate_eeg,
    generate_sine,
)

# ---------------------------------------------------------------------------
# Step 1: Individual generator functions
# ---------------------------------------------------------------------------


class TestGenerateSine:
    """Tests for generate_sine()."""

    def test_correct_length(self) -> None:
        """Output length = duration_s * fs."""
        sig = generate_sine(2.0, 250.0)
        assert len(sig) == 500

    def test_correct_frequency(self) -> None:
        """Dominant FFT peak matches the requested frequency."""
        fs = 1000.0
        freq = 10.0
        sig = generate_sine(1.0, fs, freq=freq)
        fft_freqs = np.fft.rfftfreq(len(sig), d=1.0 / fs)
        magnitudes = np.abs(np.fft.rfft(sig))
        # Exclude DC (index 0)
        peak_freq = fft_freqs[1:][np.argmax(magnitudes[1:])]
        assert abs(peak_freq - freq) < 1.0  # within 1 Hz bin

    def test_correct_amplitude(self) -> None:
        """Peak amplitude matches requested amplitude."""
        sig = generate_sine(1.0, 1000.0, amplitude=3.5)
        assert abs(np.max(np.abs(sig)) - 3.5) < 0.01

    def test_phase_offset(self) -> None:
        """Phase=π/2 shifts sine to cosine (starts at amplitude)."""
        sig = generate_sine(1.0, 1000.0, freq=1.0, phase=np.pi / 2)
        # At t=0, sin(π/2) = 1.0
        assert abs(sig[0] - 1.0) < 0.01

    def test_zero_duration(self) -> None:
        """Zero duration produces empty array."""
        sig = generate_sine(0.0, 250.0)
        assert len(sig) == 0


class TestGenerateECG:
    """Tests for generate_ecg()."""

    def test_correct_length(self) -> None:
        sig = generate_ecg(2.0, 250.0)
        assert len(sig) == 500

    def test_r_peak_is_dominant(self) -> None:
        """R wave (amplitude 1.0) should be the tallest peak."""
        sig = generate_ecg(2.0, 500.0, heart_rate=60.0)
        assert np.max(sig) > 0.8  # R wave amplitude is 1.0

    def test_heart_rate_changes_period(self) -> None:
        """Higher heart rate → more beats → more R-peaks."""
        fs = 500.0
        sig_60 = generate_ecg(4.0, fs, heart_rate=60.0)
        sig_120 = generate_ecg(4.0, fs, heart_rate=120.0)
        # Count R-peaks: find local maxima above 0.5
        from scipy.signal import find_peaks  # type: ignore[import-untyped]

        peaks_60, _ = find_peaks(sig_60, height=0.5, distance=int(fs * 0.3))
        peaks_120, _ = find_peaks(sig_120, height=0.5, distance=int(fs * 0.3))
        # 120 BPM should have ~2x as many peaks as 60 BPM
        assert len(peaks_120) > len(peaks_60)

    def test_p_multiplier_scales_p_wave(self) -> None:
        """p_multiplier=0 should suppress the P wave."""
        sig_normal = generate_ecg(2.0, 500.0, p_multiplier=1.0)
        sig_no_p = generate_ecg(2.0, 500.0, p_multiplier=0.0)
        # Signals should differ (P wave is suppressed)
        assert not np.array_equal(sig_normal, sig_no_p)
        # With p_multiplier=0, no P wave contribution
        # The max should still be ~1.0 (R peak), but the signal shape differs
        assert np.max(sig_no_p) > 0.8

    def test_baseline_wander(self) -> None:
        """Baseline wander adds a low-frequency component."""
        sig_clean = generate_ecg(4.0, 250.0, baseline_wander_amplitude=0.0)
        sig_wander = generate_ecg(4.0, 250.0, baseline_wander_amplitude=0.5)
        diff = sig_wander - sig_clean
        # The difference should be a 0.5Hz sine with amplitude 0.5
        assert abs(np.max(np.abs(diff)) - 0.5) < 0.01

    def test_deterministic(self) -> None:
        """ECG is deterministic (no RNG)."""
        a = generate_ecg(1.0, 250.0)
        b = generate_ecg(1.0, 250.0)
        assert np.array_equal(a, b)


class TestGenerateEEG:
    """Tests for generate_eeg()."""

    def test_correct_length(self) -> None:
        sig = generate_eeg(2.0, 250.0, rng=np.random.default_rng(0))
        assert len(sig) == 500

    def test_default_bands(self) -> None:
        """Default bands produce a non-zero signal."""
        sig = generate_eeg(1.0, 250.0, rng=np.random.default_rng(0))
        assert np.std(sig) > 0.1

    def test_custom_bands(self) -> None:
        """Custom single-frequency band produces a near-sinusoidal signal."""
        sig = generate_eeg(
            1.0,
            1000.0,
            bands=[(10.0, 1.0)],
            rng=np.random.default_rng(0),
        )
        # Should have a dominant peak near 10 Hz
        fft_freqs = np.fft.rfftfreq(len(sig), d=1.0 / 1000.0)
        magnitudes = np.abs(np.fft.rfft(sig))
        peak_freq = fft_freqs[1:][np.argmax(magnitudes[1:])]
        assert abs(peak_freq - 10.0) < 2.0

    def test_epileptiform_spikes(self) -> None:
        """Spike injection increases signal amplitude."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        sig_no_spikes = generate_eeg(
            2.0,
            250.0,
            spike_rate_hz=0.0,
            rng=rng1,
        )
        sig_spikes = generate_eeg(
            2.0,
            250.0,
            spike_rate_hz=50.0,
            rng=rng2,
        )
        # With spikes, max amplitude should be noticeably larger
        assert np.max(np.abs(sig_spikes)) > np.max(np.abs(sig_no_spikes))

    def test_reproducibility_with_seed(self) -> None:
        """Same RNG seed → identical output."""
        a = generate_eeg(1.0, 250.0, rng=np.random.default_rng(7))
        b = generate_eeg(1.0, 250.0, rng=np.random.default_rng(7))
        assert np.array_equal(a, b)

    def test_different_seeds_diverge(self) -> None:
        """Different seeds → different outputs."""
        a = generate_eeg(1.0, 250.0, rng=np.random.default_rng(1))
        b = generate_eeg(1.0, 250.0, rng=np.random.default_rng(2))
        assert not np.array_equal(a, b)


class TestAddFlicker:
    """Tests for add_flicker()."""

    def test_output_shape_preserved(self) -> None:
        sig = np.zeros(500)
        out = add_flicker(sig, 250.0, 50.0, 1.0)
        assert out.shape == sig.shape

    def test_correct_frequency_via_fft(self) -> None:
        """Flicker at 50 Hz should show an FFT peak at 50 Hz."""
        fs = 1000.0
        sig = np.zeros(int(fs))  # 1 second of silence
        out = add_flicker(sig, fs, 50.0, 1.0)
        fft_freqs = np.fft.rfftfreq(len(out), d=1.0 / fs)
        magnitudes = np.abs(np.fft.rfft(out))
        peak_freq = fft_freqs[1:][np.argmax(magnitudes[1:])]
        assert abs(peak_freq - 50.0) < 1.0

    def test_amplitude_scaling(self) -> None:
        """Flicker amplitude matches the requested value."""
        # Use fs=1000 so 50Hz sine hits its peak exactly on samples
        sig = np.zeros(1000)
        out = add_flicker(sig, 1000.0, 50.0, 2.5)
        assert abs(np.max(np.abs(out)) - 2.5) < 0.01

    def test_zero_amplitude_is_identity(self) -> None:
        """Zero amplitude flicker leaves signal unchanged."""
        sig = np.random.default_rng(0).random(100)
        out = add_flicker(sig, 250.0, 50.0, 0.0)
        assert np.array_equal(sig, out)


class TestAddGaussianNoise:
    """Tests for add_gaussian_noise()."""

    def test_output_shape_preserved(self) -> None:
        sig = np.sin(np.linspace(0, 10, 500))
        out = add_gaussian_noise(sig, 20.0, rng=np.random.default_rng(0))
        assert out.shape == sig.shape

    def test_snr_within_tolerance(self) -> None:
        """Achieved SNR should be close to the target SNR."""
        rng = np.random.default_rng(42)
        sig = np.sin(np.linspace(0, 20 * np.pi, 10000))
        target_snr = 10.0
        noisy = add_gaussian_noise(sig, target_snr, rng=rng)

        noise = noisy - sig
        sig_power = np.mean(sig**2)
        noise_power = np.mean(noise**2)
        actual_snr = 10 * np.log10(sig_power / noise_power)
        # Should be within ~1 dB of target for large signals
        assert abs(actual_snr - target_snr) < 1.5

    def test_reproducibility_with_seed(self) -> None:
        sig = np.ones(100)
        a = add_gaussian_noise(sig, 10.0, rng=np.random.default_rng(0))
        b = add_gaussian_noise(sig, 10.0, rng=np.random.default_rng(0))
        assert np.array_equal(a, b)

    def test_zero_signal_returns_copy(self) -> None:
        """Zero-power signal returns a copy (no noise added)."""
        sig = np.zeros(100)
        out = add_gaussian_noise(sig, 10.0)
        assert np.array_equal(out, sig)
        # Must be a copy, not the same object
        assert out is not sig

    def test_high_snr_means_low_noise(self) -> None:
        """Very high SNR should produce negligible noise."""
        sig = np.sin(np.linspace(0, 10, 1000))
        noisy = add_gaussian_noise(sig, 60.0, rng=np.random.default_rng(0))
        assert np.allclose(sig, noisy, atol=0.01)

    def test_low_snr_means_high_noise(self) -> None:
        """Very low SNR should produce significant noise."""
        sig = np.sin(np.linspace(0, 10, 1000))
        noisy = add_gaussian_noise(sig, 0.0, rng=np.random.default_rng(0))
        # At 0 dB, noise power ≈ signal power → large differences
        assert np.mean((noisy - sig) ** 2) > 0.1


class TestAddInterference:
    """Tests for add_interference() — the R3 fix for the flicker SNR confound."""

    @pytest.mark.parametrize("kind", ["gaussian", "flicker", "mixed"])
    @pytest.mark.parametrize("snr_db", [-20.0, 0.0, 17.5, 40.0])
    def test_input_snr_is_exact(self, kind: InterferenceKind, snr_db: float) -> None:
        clean = generate_ecg(duration_s=4.0, fs=250.0)
        noisy = add_interference(clean, snr_db, kind=kind, rng=np.random.default_rng(0))
        assert calculate_snr(clean, noisy) == pytest.approx(snr_db, abs=1e-9)

    def test_high_snr_is_reachable_on_the_axis_the_old_pipeline_capped(self) -> None:
        """
        SignalGenerator with 0.5-amplitude flicker cannot exceed about -6 dB on
        ECG against the clean signal, whatever `snr_db` asks for.
        """
        clean, capped = SignalGenerator(fs=250.0, seed=0).generate(
            "ecg", 4.0, flicker_freq=50.0, flicker_amplitude=0.5, snr_db=40.0
        )
        assert calculate_snr(clean, capped) < 0.0
        fixed = add_interference(
            clean, 40.0, kind="mixed", rng=np.random.default_rng(0)
        )
        assert calculate_snr(clean, fixed) == pytest.approx(40.0, abs=1e-9)

    def test_flicker_is_a_sinusoid_at_the_requested_frequency(self) -> None:
        fs, freq = 250.0, 50.0
        clean = np.zeros(1000) + np.sin(np.linspace(0, 8 * np.pi, 1000))
        noisy = add_interference(
            clean,
            0.0,
            kind="flicker",
            fs=fs,
            flicker_freq=freq,
            rng=np.random.default_rng(1),
        )
        spectrum = np.abs(np.fft.rfft(noisy - clean))
        peak_hz = np.fft.rfftfreq(clean.size, d=1.0 / fs)[int(np.argmax(spectrum))]
        assert peak_hz == pytest.approx(freq, abs=fs / clean.size)

    def test_mixed_interference_splits_power_between_its_parts(self) -> None:
        fs = 250.0
        clean = generate_ecg(duration_s=8.0, fs=fs)
        noisy = add_interference(
            clean, 0.0, kind="mixed", fs=fs, rng=np.random.default_rng(2)
        )
        spectrum = np.abs(np.fft.rfft(noisy - clean)) ** 2
        freqs = np.fft.rfftfreq(clean.size, d=1.0 / fs)
        tone = spectrum[np.abs(freqs - 50.0) <= 0.5].sum() / spectrum.sum()
        assert 0.4 < tone < 0.6, f"flicker carries {tone:.2f} of the power"

    def test_reproducible_and_phase_varies_with_the_seed(self) -> None:
        clean = generate_ecg(duration_s=2.0, fs=250.0)
        a = add_interference(clean, 5.0, "flicker", rng=np.random.default_rng(3))
        b = add_interference(clean, 5.0, "flicker", rng=np.random.default_rng(3))
        c = add_interference(clean, 5.0, "flicker", rng=np.random.default_rng(4))
        np.testing.assert_array_equal(a, b)
        assert not np.allclose(a, c)

    def test_constant_signal_returns_an_unmodified_copy(self) -> None:
        clean = np.full(100, 3.0)
        out = add_interference(clean, 10.0)
        np.testing.assert_array_equal(out, clean)
        assert out is not clean

    def test_bad_arguments_raise(self) -> None:
        clean = generate_ecg(duration_s=1.0, fs=250.0)
        with pytest.raises(ValueError, match="finite"):
            add_interference(clean, float("inf"))
        with pytest.raises(ValueError, match="Unknown interference kind"):
            add_interference(clean, 0.0, kind="pink")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Step 2: SignalGenerator pipeline class
# ---------------------------------------------------------------------------


class TestSignalGenerator:
    """Tests for the SignalGenerator unified pipeline."""

    def test_ecg_clean_output(self) -> None:
        """ECG with no noise returns identical clean and noisy."""
        gen = SignalGenerator(fs=250.0, seed=0)
        clean, noisy = gen.generate("ecg", duration_s=1.0)
        assert np.array_equal(clean, noisy)
        assert len(clean) == 250

    def test_eeg_clean_output(self) -> None:
        """EEG with no noise returns identical clean and noisy."""
        gen = SignalGenerator(fs=250.0, seed=0)
        clean, noisy = gen.generate("eeg", duration_s=1.0)
        assert np.array_equal(clean, noisy)

    def test_sine_clean_output(self) -> None:
        """Sine with no noise returns identical clean and noisy."""
        gen = SignalGenerator(fs=250.0, seed=0)
        clean, noisy = gen.generate("sine", duration_s=1.0)
        assert np.array_equal(clean, noisy)

    def test_with_flicker_only(self) -> None:
        """Flicker injection makes noisy differ from clean."""
        gen = SignalGenerator(fs=1000.0, seed=0)
        clean, noisy = gen.generate(
            "ecg",
            duration_s=1.0,
            flicker_freq=50.0,
            flicker_amplitude=0.5,
        )
        assert not np.array_equal(clean, noisy)
        # The difference should be a sinusoidal flicker
        diff = noisy - clean
        assert abs(np.max(np.abs(diff)) - 0.5) < 0.01

    def test_with_gaussian_only(self) -> None:
        """Gaussian noise injection makes noisy differ from clean."""
        gen = SignalGenerator(fs=250.0, seed=0)
        clean, noisy = gen.generate(
            "sine",
            duration_s=1.0,
            snr_db=10.0,
        )
        assert not np.array_equal(clean, noisy)

    def test_with_combined_noise(self) -> None:
        """Both flicker and Gaussian applied together."""
        gen = SignalGenerator(fs=250.0, seed=0)
        clean, noisy = gen.generate(
            "ecg",
            duration_s=1.0,
            flicker_freq=50.0,
            flicker_amplitude=0.3,
            snr_db=10.0,
        )
        assert not np.array_equal(clean, noisy)
        # Difference should be larger than flicker alone (noise added)
        diff_std = np.std(noisy - clean)
        assert diff_std > 0.1

    def test_flicker_disabled_when_amplitude_zero(self) -> None:
        """Flicker with amplitude=0 should not change the signal."""
        gen = SignalGenerator(fs=250.0, seed=0)
        clean, noisy = gen.generate(
            "ecg",
            duration_s=1.0,
            flicker_freq=50.0,
            flicker_amplitude=0.0,
        )
        assert np.array_equal(clean, noisy)

    def test_reproducibility(self) -> None:
        """Same seed → identical (clean, noisy) outputs."""

        def run(seed: int) -> tuple[np.ndarray, np.ndarray]:
            gen = SignalGenerator(fs=250.0, seed=seed)
            return gen.generate(
                "eeg",
                duration_s=1.0,
                flicker_freq=50.0,
                flicker_amplitude=0.5,
                snr_db=10.0,
            )

        c1, n1 = run(42)
        c2, n2 = run(42)
        assert np.array_equal(c1, c2)
        assert np.array_equal(n1, n2)

    def test_different_seeds_diverge(self) -> None:
        """Different seeds → different outputs."""

        def run(seed: int) -> np.ndarray:
            gen = SignalGenerator(fs=250.0, seed=seed)
            _, noisy = gen.generate(
                "eeg",
                duration_s=1.0,
                snr_db=10.0,
            )
            return noisy

        assert not np.array_equal(run(1), run(2))

    def test_invalid_signal_type(self) -> None:
        """Unknown signal type raises ValueError."""
        gen = SignalGenerator(fs=250.0, seed=0)
        with pytest.raises(ValueError, match="Unknown signal type"):
            gen.generate("ppg", duration_s=1.0)  # type: ignore[arg-type]

    def test_kwargs_forwarded_to_ecg(self) -> None:
        """Extra kwargs are forwarded to generate_ecg."""
        gen = SignalGenerator(fs=250.0, seed=0)
        clean_default, _ = gen.generate("ecg", duration_s=2.0)
        gen2 = SignalGenerator(fs=250.0, seed=0)
        clean_custom, _ = gen2.generate(
            "ecg",
            duration_s=2.0,
            heart_rate=120.0,
        )
        # Different heart rate → different signal
        assert not np.array_equal(clean_default, clean_custom)

    def test_kwargs_forwarded_to_sine(self) -> None:
        """Extra kwargs are forwarded to generate_sine."""
        gen = SignalGenerator(fs=250.0, seed=0)
        clean, _ = gen.generate(
            "sine",
            duration_s=1.0,
            freq=5.0,
            amplitude=2.0,
        )
        assert abs(np.max(np.abs(clean)) - 2.0) < 0.01
