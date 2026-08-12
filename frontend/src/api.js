// Keep browser requests on the same origin by default. Vite proxies /api during
// development; deployed installations can set VITE_API_BASE for a separate API.
const configuredBase = import.meta.env.VITE_API_BASE || '';
export const API_BASE = configuredBase.replace(/\/$/, '');

export const apiUrl = (path) => {
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  return `${API_BASE}${normalizedPath}`;
};
