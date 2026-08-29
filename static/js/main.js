/* ==========================================================================
   CIFAKE - shared front-end behaviour
   1) Dark / light theme toggle (persisted in localStorage)
   2) Scroll-reveal animation for .reveal elements
   3) Drag-and-drop upload widget with preview + submit spinner
   4) Private browser prediction history (persisted in localStorage)
   ========================================================================== */

const HISTORY_STORAGE_KEY = "cifake-prediction-history";
const HISTORY_LIMIT = 20;

document.addEventListener("DOMContentLoaded", () => {
  initThemeToggle();
  initScrollReveal();
  initUploadWidget();
  animateProbabilityBars();
  saveResultToHistory();
  initHistoryPage();
});

/* ------------------------------ Theme toggle ------------------------------ */
function initThemeToggle() {
  const root = document.documentElement;
  const toggleBtn = document.getElementById("themeToggle");
  const stored = localStorage.getItem("cifake-theme");
  const initial = stored || "dark";
  root.setAttribute("data-theme", initial);
  updateToggleIcon(initial);

  if (!toggleBtn) return;
  toggleBtn.addEventListener("click", () => {
    const current = root.getAttribute("data-theme");
    const next = current === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    localStorage.setItem("cifake-theme", next);
    updateToggleIcon(next);
  });
}

function updateToggleIcon(theme) {
  const toggleBtn = document.getElementById("themeToggle");
  if (!toggleBtn) return;
  toggleBtn.innerHTML =
    theme === "dark" ? '<i class="bi bi-sun-fill"></i>' : '<i class="bi bi-moon-stars-fill"></i>';
}

/* ------------------------------ Scroll reveal ------------------------------ */
function initScrollReveal() {
  const elements = document.querySelectorAll(".reveal");
  if (!("IntersectionObserver" in window) || elements.length === 0) {
    elements.forEach((el) => el.classList.add("is-visible"));
    return;
  }
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.15 }
  );
  elements.forEach((el) => observer.observe(el));
}

/* ------------------------------ Upload widget ------------------------------ */
function initUploadWidget() {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const form = document.getElementById("uploadForm");
  if (!dropzone || !fileInput || !form) return;

  const previewWrap = document.getElementById("previewWrap");
  const previewImg = document.getElementById("previewImg");
  const previewName = document.getElementById("previewName");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const spinner = document.getElementById("spinnerOverlay");
  const ALLOWED = ["image/jpeg", "image/jpg", "image/png"];

  const openPicker = () => fileInput.click();
  dropzone.addEventListener("click", openPicker);
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") openPicker();
  });

  ["dragenter", "dragover"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    })
  );
  ["dragleave", "drop"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    })
  );
  dropzone.addEventListener("drop", (e) => {
    const files = e.dataTransfer.files;
    if (files && files.length) {
      fileInput.files = files;
      handleFile(files[0]);
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files[0]) {
      handleFile(fileInput.files[0]);
    }
  });

  function handleFile(file) {
    if (!ALLOWED.includes(file.type)) {
      alert("Only JPG, JPEG, and PNG images are supported.");
      fileInput.value = "";
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      previewName.textContent = `${file.name} - ${(file.size / 1024).toFixed(1)} KB`;
      previewWrap.classList.add("active");
      analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  form.addEventListener("submit", () => {
    if (!fileInput.files || !fileInput.files[0]) return;
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Analyzing...';
    if (spinner) spinner.classList.add("active");
  });
}

/* ------------------------------ Probability bars ------------------------------ */
function animateProbabilityBars() {
  document.querySelectorAll(".prob-fill[data-target]").forEach((bar) => {
    const target = bar.getAttribute("data-target");
    requestAnimationFrame(() => {
      bar.style.width = `${target}%`;
    });
  });
}

/* ------------------------------ Local history ------------------------------ */
function getHistoryRecords() {
  try {
    const records = JSON.parse(localStorage.getItem(HISTORY_STORAGE_KEY) || "[]");
    return Array.isArray(records) ? records : [];
  } catch (error) {
    return [];
  }
}

function setHistoryRecords(records) {
  localStorage.setItem(
    HISTORY_STORAGE_KEY,
    JSON.stringify(sortHistoryRecords(records).slice(0, HISTORY_LIMIT))
  );
}

function saveResultToHistory() {
  const dataEl = document.getElementById("prediction-history-record");
  if (!dataEl) return;

  try {
    const record = JSON.parse(dataEl.textContent);
    if (!record || !record.id) return;
    const withoutCurrent = getHistoryRecords().filter((item) => item.id !== record.id);
    setHistoryRecords([record, ...withoutCurrent]);
  } catch (error) {
    console.warn("Unable to save prediction history.", error);
  }
}

function initHistoryPage() {
  const tableBody = document.getElementById("historyTableBody");
  const panel = document.getElementById("historyPanel");
  const emptyState = document.getElementById("historyEmptyState");
  const clearBtn = document.getElementById("clearHistoryBtn");
  if (!tableBody || !panel || !emptyState || !clearBtn) return;

  renderHistory();

  clearBtn.addEventListener("click", () => {
    const confirmed = window.confirm("Clear all prediction history stored in this browser?");
    if (!confirmed) return;
    setHistoryRecords([]);
    renderHistory();
  });

  tableBody.addEventListener("click", (event) => {
    const deleteBtn = event.target.closest("[data-delete-history-id]");
    if (!deleteBtn) return;
    const id = deleteBtn.getAttribute("data-delete-history-id");
    setHistoryRecords(getHistoryRecords().filter((record) => record.id !== id));
    renderHistory();
  });

  function renderHistory() {
    const records = getHistoryRecords();
    const hasRecords = records.length > 0;
    panel.classList.toggle("d-none", !hasRecords);
    emptyState.classList.toggle("d-none", hasRecords);
    clearBtn.classList.toggle("d-none", !hasRecords);
    tableBody.innerHTML = "";

    records.forEach((record, index) => {
      const row = document.createElement("tr");
      row.style.opacity = "0";
      row.style.transform = "translateY(8px)";
      row.style.transition = "opacity 0.22s ease, transform 0.22s ease";
      row.innerHTML = historyRowTemplate(record);
      tableBody.appendChild(row);
      requestAnimationFrame(() => {
        window.setTimeout(() => {
          row.style.opacity = "1";
          row.style.transform = "translateY(0)";
        }, index * 25);
      });
    });
  }
}

function historyRowTemplate(record) {
  const label = record.predictionLabel === "REAL" ? "REAL" : "FAKE";
  const badgeClass = label === "REAL" ? "badge-real" : "badge-fake";
  const confidence = formatPercent(record.confidence);
  const reportLink = record.reportDownloadUrl
    ? `<a href="${escapeAttribute(record.reportDownloadUrl)}" class="btn btn-sm btn-download" title="Download PDF report" aria-label="Download PDF report">
        <i class="bi bi-file-earmark-pdf"></i>
      </a>`
    : "";

  return `
    <td><img src="${escapeAttribute(record.uploadedImageUrl || "")}" class="history-thumb" alt="${escapeAttribute(record.originalFilename || "Uploaded image")}"></td>
    <td class="mono small">${escapeHtml(record.originalFilename || "Uploaded image")}</td>
    <td>
      <span class="badge ${badgeClass} px-3 py-2">${label}</span>
    </td>
    <td class="mono">${confidence}</td>
    <td class="mono small text-muted">${escapeHtml(record.timestamp || "")}</td>
    <td class="text-end">
      <div class="d-inline-flex flex-wrap justify-content-end gap-2">
        <a href="${escapeAttribute(record.resultUrl || "#")}" class="btn btn-sm btn-outline-ghost">
          <i class="bi bi-eye me-1"></i>View
        </a>
        ${reportLink}
        <button class="btn btn-sm btn-outline-ghost" type="button" data-delete-history-id="${escapeAttribute(record.id)}" aria-label="Delete history record">
          <i class="bi bi-trash3"></i>
        </button>
      </div>
    </td>
  `;
}

function formatPercent(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "0.00%";
  return `${(number * 100).toFixed(2)}%`;
}

function sortHistoryRecords(records) {
  return [...records].sort((a, b) => timestampValue(b.timestamp) - timestampValue(a.timestamp));
}

function timestampValue(timestamp) {
  if (!timestamp) return 0;
  const parsed = Date.parse(String(timestamp).replace(" ", "T"));
  return Number.isFinite(parsed) ? parsed : 0;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function escapeAttribute(value) {
  return escapeHtml(value).replaceAll("`", "&#096;");
}
