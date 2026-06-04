import type { 
  RecordSummary, 
  SegmentResponse, 
  SimulateRequest, 
  SimulateResponse, 
  UploadResponse 
} from './types';

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000/api';

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = 'ApiError';
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, options);
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new ApiError(response.status, error.detail || response.statusText);
  }
  return response.json();
}

export const listRecords = () => request<RecordSummary[]>('/records');

export const getRecordSegment = (id: string, duration_s: number) => 
  request<SegmentResponse>(`/records/${id}/segment?duration_s=${duration_s}`);

export const uploadCsv = async (file: File, fs: number): Promise<UploadResponse> => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('fs', fs.toString());
  
  return request<UploadResponse>('/upload', {
    method: 'POST',
    body: formData,
  });
};

export const simulate = (req: SimulateRequest) => 
  request<SimulateResponse>('/simulate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });

export const exportCsvUrl = (simulationId: string) => 
  `${API_BASE}/export/${simulationId}.csv`;
