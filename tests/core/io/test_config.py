import json
from pathlib import Path

import pytest
from neuroflux_core.io.config import LIFConfig, NetworkConfig, STDPConfig
from pydantic import ValidationError


def test_lif_config_defaults() -> None:
    config = LIFConfig()
    assert config.v_rest == -70.0
    assert config.v_thresh == -55.0
    assert config.v_reset == -80.0
    assert config.tau_m == 20.0
    assert config.t_ref == 2.0


def test_lif_config_custom() -> None:
    config = LIFConfig(v_rest=-65.0, v_thresh=-50.0)
    assert config.v_rest == -65.0
    assert config.v_thresh == -50.0


def test_stdp_config_defaults() -> None:
    config = STDPConfig()
    assert config.a_plus == 0.01
    assert config.a_minus == 0.012
    assert config.tau_plus == 20.0
    assert config.tau_minus == 20.0
    assert config.w_min == 0.0
    assert config.w_max == 1.0


def test_network_config_defaults() -> None:
    config = NetworkConfig()
    assert config.layers == [32, 16, 8]
    assert isinstance(config.neuron, LIFConfig)
    assert isinstance(config.plasticity, STDPConfig)


def test_config_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        LIFConfig(unknown_field=123)  # type: ignore


def test_network_config_from_yaml(tmp_path: Path) -> None:
    yaml_content = """
layers: [64, 32]
neuron:
  tau_m: 15.0
plasticity:
  a_plus: 0.02
"""
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(yaml_content)

    config = NetworkConfig.from_yaml(yaml_file)
    assert config.layers == [64, 32]
    assert config.neuron.tau_m == 15.0
    assert config.neuron.v_rest == -70.0  # default
    assert config.plasticity.a_plus == 0.02
    assert config.plasticity.a_minus == 0.012  # default


def test_network_config_from_json(tmp_path: Path) -> None:
    json_data = {
        "layers": [10, 5],
        "neuron": {"v_thresh": -40.0},
        "plasticity": {"w_max": 2.0},
    }
    json_file = tmp_path / "config.json"
    json_file.write_text(json.dumps(json_data))

    config = NetworkConfig.from_json(json_file)
    assert config.layers == [10, 5]
    assert config.neuron.v_thresh == -40.0
    assert config.plasticity.w_max == 2.0


def test_network_config_from_dict() -> None:
    data = {
        "layers": [5, 2],
        "neuron": {"tau_m": 12.0},
    }
    config = NetworkConfig.from_dict(data)
    assert config.layers == [5, 2]
    assert config.neuron.tau_m == 12.0
    assert config.plasticity.a_plus == 0.01  # default
