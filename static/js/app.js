/* ============================================================
   XRayGuard – Frontend JavaScript
   Handles: upload, preview, API calls, report rendering
   ============================================================ */

document.addEventListener('DOMContentLoaded', () => {
    const uploadZone  = document.getElementById('uploadZone');
    const fileInput   = document.getElementById('fileInput');
    const previewImg  = document.getElementById('previewImg');
    const placeholder = document.getElementById('uploadPlaceholder');
    const btnGenerate = document.getElementById('btnGenerate');
    const demoBanner  = document.getElementById('demoBanner');

    let selectedFile = null;

    // ---- Check server health on load ----
    checkHealth();

    // ---- Upload zone click / drag ----
    uploadZone.addEventListener('click', () => fileInput.click());

    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('drag-over');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('drag-over');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            handleFile(fileInput.files[0]);
        }
    });

    const loadSampleLink = document.getElementById('loadSampleLink');
    if (loadSampleLink) {
        loadSampleLink.addEventListener('click', async (e) => {
            e.preventDefault();
            try {
                const res = await fetch('/static/samples/sample_chest_xray.png');
                const blob = await res.blob();
                const file = new File([blob], 'sample_chest_xray.png', { type: 'image/png' });
                handleFile(file);
            } catch (err) {
                console.error('Failed to load sample image:', err);
                showError('Could not load sample image.');
            }
        });
    }

    btnGenerate.addEventListener('click', generateReport);

    // ---- Handle file selection ----
    function handleFile(file) {
        if (!file) return;

        // Validate size (25 MB)
        if (file.size > 25 * 1024 * 1024) {
            showError('File is too large. Maximum size is 25 MB.');
            return;
        }

        selectedFile = file;

        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            previewImg.style.display = 'block';
            placeholder.style.display = 'none';
            uploadZone.classList.add('has-image');
        };
        reader.readAsDataURL(file);

        btnGenerate.disabled = false;

        // Reset report card state
        document.getElementById('reportContent').style.display = 'none';
        const rp = document.getElementById('reportPlaceholder');
        rp.style.display = 'flex';
        rp.innerHTML = `
            <span class="material-symbols-outlined">description</span>
            <p>Click "Generate report" to run AI analysis on this X-ray.</p>
        `;
        document.getElementById('exportBar').style.display = 'none';
    }

    // ---- Generate report ----
    async function generateReport() {
        if (!selectedFile) {
            showError('Please select or drag an X-ray image first.');
            return;
        }

        const btnText    = btnGenerate.querySelector('.btn-text');
        const btnSpinner = btnGenerate.querySelector('.btn-spinner');
        btnGenerate.classList.add('loading');
        btnGenerate.disabled = true;
        btnText.style.display = 'none';
        btnSpinner.style.display = 'flex';

        // Show loading state in report box
        const reportPlaceholder = document.getElementById('reportPlaceholder');
        reportPlaceholder.style.display = 'flex';
        reportPlaceholder.innerHTML = `
            <div class="loading-state" style="text-align:center; padding: 40px 10px;">
                <svg class="spinner" viewBox="0 0 24 24" style="width:40px;height:40px;margin-bottom:14px;color:#2b7a78;">
                    <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="3" fill="none"
                            stroke-dasharray="31.4" stroke-dashoffset="10"/>
                </svg>
                <p style="font-size:15px; font-weight:600; color:#1f2937;">Analyzing chest X-ray...</p>
                <p style="font-size:13px; color:#6b7280; margin-top:6px;">
                    DenseNet121 inference · GradCAM attention · Report synthesis
                </p>
            </div>
        `;
        document.getElementById('reportContent').style.display = 'none';
        document.getElementById('exportBar').style.display = 'none';

        try {
            const formData = new FormData();
            formData.append('image', selectedFile);

            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                let errorMsg = `Server error ${response.status}`;
                try {
                    const err = await response.json();
                    if (err.error) errorMsg = err.error;
                } catch (_) {}
                throw new Error(errorMsg);
            }

            const data = await response.json();
            if (!data.success) {
                throw new Error(data.error || 'Prediction was unsuccessful.');
            }

            renderReport(data);

        } catch (err) {
            console.error('Prediction error:', err);
            showError(err.message || 'Failed to connect to the backend server.');
            reportPlaceholder.style.display = 'flex';
            reportPlaceholder.innerHTML = `
                <span class="material-symbols-outlined" style="color:#ef4444; font-size:48px; margin-bottom:8px;">error_outline</span>
                <p style="color:#b91c1c; font-weight:600; margin-bottom:4px;">Report generation failed</p>
                <p style="font-size:13px; color:#6b7280; max-width:320px;">${err.message || 'Please check that the server is running and try again.'}</p>
                <button onclick="document.getElementById('btnGenerate').click()" style="margin-top:12px; padding:6px 14px; background:#f3f4f6; border:1px solid #d1d5db; border-radius:6px; cursor:pointer; font-size:13px;">Retry</button>
            `;
        } finally {
            btnGenerate.classList.remove('loading');
            btnGenerate.disabled = false;
            btnText.style.display = 'inline';
            btnSpinner.style.display = 'none';
        }
    }

    // ---- Render report ----
    function renderReport(data) {
        document.getElementById('reportPlaceholder').style.display = 'none';
        document.getElementById('reportContent').style.display = 'block';
        document.getElementById('exportBar').style.display = 'flex';

        // Findings
        document.getElementById('findingsText').textContent =
            data.report?.findings || 'No findings available.';

        // Impression
        document.getElementById('impressionText').textContent =
            data.report?.impression || 'No impression available.';

        // Confidence scores
        renderConfidence(data.probabilities || {});

        // Heatmap
        if (data.heatmap) {
            document.getElementById('heatmapSection').style.display = 'block';
            document.getElementById('heatmapImg').src =
                'data:image/png;base64,' + data.heatmap;
        } else {
            document.getElementById('heatmapSection').style.display = 'none';
        }

        // Tiers
        renderTiers(data.tiers || {});

        // Metadata
        document.getElementById('reportTime').textContent =
            data.time_ms ? `Inference time: ${data.time_ms} ms` : '';
        document.getElementById('reportMode').textContent =
            data.demo_mode ? '⚠ Demo mode (ImageNet weights)' : '✓ Trained NIH model';

        // Show demo banner if needed
        if (data.demo_mode) {
            demoBanner.style.display = 'flex';
        }
    }

    // ---- Confidence bars ----
    function renderConfidence(probs) {
        const grid = document.getElementById('confidenceGrid');
        grid.innerHTML = '';

        // Sort by probability descending
        const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1]);

        sorted.forEach(([label, prob]) => {
            const pct = (prob * 100).toFixed(1);
            let cls = 'conf-low';
            if (prob >= 0.5) cls = 'conf-high';
            else if (prob >= 0.3) cls = 'conf-medium';

            const item = document.createElement('div');
            item.className = `confidence-item ${cls}`;
            item.innerHTML = `
                <span class="confidence-label">${label}</span>
                <div class="confidence-bar-bg">
                    <div class="confidence-bar" style="width: ${pct}%"></div>
                </div>
                <span class="confidence-value">${pct}%</span>
            `;
            grid.appendChild(item);
        });
    }

    // ---- Tiers ----
    function renderTiers(tiers) {
        const container = document.getElementById('tiersContainer');
        container.innerHTML = '';

        const sections = [
            { key: 'present',  label: 'Present (High Confidence)',  cls: 'tier-present'  },
            { key: 'possible', label: 'Possible (Moderate Confidence)', cls: 'tier-possible' },
            { key: 'absent',   label: 'Absent / Unremarkable',   cls: 'tier-absent'   },
        ];

        sections.forEach(({ key, label, cls }) => {
            const items = tiers[key] || [];
            if (items.length === 0) return;

            const group = document.createElement('div');
            group.className = `tier-group ${cls}`;
            group.innerHTML = `
                <span class="tier-label">${label}</span>
                <div class="tier-items">
                    ${items.map(item =>
                        `<span class="tier-tag">${item.finding} (${item.confidence}%)</span>`
                    ).join('')}
                </div>
            `;
            container.appendChild(group);
        });

        if (container.children.length === 0) {
            container.innerHTML = '<p style="color:#9ca3af;font-size:13px;">No tier data available.</p>';
        }
    }

    // ---- Health check ----
    async function checkHealth() {
        try {
            const res = await fetch('/api/health');
            const data = await res.json();
            if (data.demo_mode) {
                demoBanner.style.display = 'flex';
            }
        } catch (e) {
            console.log('Server health check failed (may still be starting).');
        }
    }

    // ---- Error toast ----
    function showError(msg) {
        const toast = document.createElement('div');
        toast.className = 'error-toast';
        toast.textContent = msg;
        document.body.appendChild(toast);
        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateY(20px)';
            toast.style.transition = '0.3s ease';
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    }
});

// ---- Global functions (export bar) ----

function copyReport() {
    const findings   = document.getElementById('findingsText')?.textContent || '';
    const impression = document.getElementById('impressionText')?.textContent || '';
    const text = `FINDINGS:\n${findings}\n\nIMPRESSION:\n${impression}`;
    navigator.clipboard.writeText(text).then(() => {
        showToast('Report copied to clipboard!');
    });
}

function downloadReport() {
    const findings   = document.getElementById('findingsText')?.textContent || '';
    const impression = document.getElementById('impressionText')?.textContent || '';
    const text =
`XRAYGUARD – CHEST X-RAY REPORT
================================
Generated: ${new Date().toLocaleString()}

FINDINGS:
${findings}

IMPRESSION:
${impression}

================================
Disclaimer: This report was generated by an AI model (DenseNet121,
trained on NIH ChestX-ray14) and is intended for research use only.
All findings must be reviewed by a qualified radiologist.
`;

    const blob = new Blob([text], { type: 'text/plain' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `xrayguard_report_${Date.now()}.txt`;
    a.click();
}

function resetForm() {
    document.getElementById('previewImg').style.display = 'none';
    document.getElementById('uploadPlaceholder').style.display = 'flex';
    document.getElementById('uploadZone').classList.remove('has-image');
    document.getElementById('fileInput').value = '';
    document.getElementById('btnGenerate').disabled = true;
    document.getElementById('reportContent').style.display = 'none';
    document.getElementById('exportBar').style.display = 'none';

    const rp = document.getElementById('reportPlaceholder');
    rp.style.display = 'flex';
    rp.innerHTML = `
        <span class="material-symbols-outlined">description</span>
        <p>Report generation will be available after you upload an X-ray image.</p>
    `;
}

function showToast(msg) {
    const toast = document.createElement('div');
    toast.className = 'error-toast';
    toast.style.background = '#10b981';
    toast.textContent = msg;
    document.body.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 300);
    }, 2000);
}
