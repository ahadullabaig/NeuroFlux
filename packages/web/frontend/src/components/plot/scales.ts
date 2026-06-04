export const xScale = (t: number, duration: number, width: number, margin: number) => {
  return margin + (t / duration) * (width - 2 * margin);
};

export const yScale = (v: number, vMin: number, vMax: number, height: number, margin: number) => {
  if (vMax === vMin) return height / 2;
  return height - margin - ((v - vMin) / (vMax - vMin)) * (height - 2 * margin);
};

export const buildPath = (
  samples: number[], 
  duration: number, 
  width: number, 
  height: number, 
  margin: number,
  vMin: number,
  vMax: number
) => {
  if (samples.length === 0) return "";
  const dt = duration / (samples.length - 1);
  return samples
    .map((v, i) => {
      const x = xScale(i * dt, duration, width, margin);
      const y = yScale(v, vMin, vMax, height, margin);
      return `${i === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
};
