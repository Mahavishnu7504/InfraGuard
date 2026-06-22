const CLOUD_URL =
  import.meta.env.VITE_API_URL ||
  "https://infraguard-3yfj.onrender.com";

const LOCAL_URL =
  "http://127.0.0.1:8000";

const hostname =
  typeof window !== "undefined"
    ? window.location.hostname
    : "localhost";

const isLocalhost =
  hostname === "localhost" ||
  hostname === "127.0.0.1";

export const API_BASE =
  isLocalhost
    ? LOCAL_URL
    : CLOUD_URL;

// =====================================================
// ERROR EXTRACTION
// Handles FastAPI validation errors where `detail`
// is an array of objects, not a plain string.
// e.g. { detail: [{ loc: [...], msg: "...", type: "..." }] }
// =====================================================

function extractErrorMessage(data) {
  if (!data) return null;

  // FastAPI validation error: detail is an array
  if (Array.isArray(data.detail)) {
    return data.detail
      .map((e) => {
        const loc = Array.isArray(e.loc)
          ? e.loc.filter((s) => s !== "body").join(" → ")
          : "";
        return loc ? `${loc}: ${e.msg}` : e.msg;
      })
      .join(" | ");
  }

  // Standard FastAPI error: detail is a string
  if (typeof data.detail === "string") {
    return data.detail;
  }

  // Other shapes
  if (typeof data.message === "string") {
    return data.message;
  }

  return null;
}

// =====================================================
// SAFE FETCH ENGINE
// =====================================================

async function fetchJSON(
  url,
  options = {}
) {

  try {

    const response =
      await fetch(url, {

        ...options,

        headers: {
          ...(options.headers || {})
        }
      });

    let data = {};

    const contentType =
      response.headers.get("content-type") || "";

    if (contentType.includes("application/json")) {
      data = await response.json();
    }

    if (!response.ok) {
      const message =
        extractErrorMessage(data) ||
        `Request failed (${response.status})`;

      throw new Error(message);
    }

    return data;

  } catch (err) {

    console.error("[API ERROR]", err);

    throw err;
  }
}

// =====================================================
// GENERIC
// =====================================================

export const apiGet = (
  path
) =>
  fetchJSON(
    `${API_BASE}${path}`
  );

export const apiPost = (
  path,
  body,
  isForm = false
) =>

  fetchJSON(

    `${API_BASE}${path}`,

    {
      method: "POST",

      body,

      headers:
        isForm
          ? {}
          : {
            "Content-Type":
              "application/json"
          }
    }
  );

// =====================================================
// QUALITY AI
// =====================================================

export const analyzeQuality =
  async (formData) => {

    return apiPost(

      "/quality/analyze",

      formData,

      true
    );
  };

// =====================================================
// REPORTS
// =====================================================

export const generateReport =
  async (payload) => {

    return apiPost(
      "/quality/report",
      JSON.stringify(payload)
    );
  };

// =====================================================
// CAMERA STREAM
// =====================================================

export const getCameraFeedUrl = (
  camId = 0
) => {

  return `${API_BASE}/safety/camera/feed?cam_id=${camId}`;
};

// =====================================================
// CAMERA CONTROLS
// =====================================================

export const startCamera = (
  camId = 0
) =>
  fetchJSON(
    `${API_BASE}/safety/camera/start?cam_id=${camId}`
  );

export const stopCamera = (
  camId = 0
) =>
  fetchJSON(
    `${API_BASE}/safety/camera/stop?cam_id=${camId}`
  );

export const getCameraStatus = (
  camId = 0
) =>
  fetchJSON(
    `${API_BASE}/safety/camera/status?cam_id=${camId}`
  );

// =====================================================
// ANALYTICS
// =====================================================

export const getAnalyticsSummary =
  () =>
    apiGet(
      "/activity/analytics/summary"
    );

export const getAlerts =
  () =>
    apiGet(
      "/activity/latest/0"
    );