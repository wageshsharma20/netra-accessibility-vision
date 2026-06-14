// static/js/script.js

// ===== THEME TOGGLE =====
(function () {
    const root = document.documentElement;
    const toggle = document.getElementById('themeToggle');
    let stored = null;
    
    try { stored = localStorage.getItem('netra-theme'); } catch (e) {}
    const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    const theme = stored || (prefersDark ? 'dark' : 'light');
    
    applyTheme(theme);

    function applyTheme(t) {
        root.setAttribute('data-theme', t);
        const isDark = t === 'dark';
        toggle.setAttribute('aria-pressed', String(isDark));
        toggle.setAttribute('aria-label', isDark ? 'Switch to light mode' : 'Switch to dark mode');
    }

    toggle.addEventListener('click', function () {
        const next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
        applyTheme(next);
        try { localStorage.setItem('netra-theme', next); } catch (e) {}
    });
})();

// ===== MACHINE LEARNING API ENGINE =====
(function () {
    const viewfinder = document.getElementById('viewfinder');
    const vfArt = document.getElementById('vfArt');
    const statusEl = document.getElementById('status');
    const resultText = document.getElementById('resultText');
    const resultSub = document.getElementById('resultSub');
    
    const btnCurrency = document.getElementById('scanCurrency');
    const formCurrency = document.getElementById('scanCurrencyForm');

    function speakAloud(text) {
        if (!('speechSynthesis' in window)) return;
        try {
            window.speechSynthesis.cancel();
            const u = new SpeechSynthesisUtterance(text);
            u.rate = 0.95; 
            u.pitch = 1; 
            u.lang = 'en-IN'; 
            window.speechSynthesis.speak(u);
        } catch (e) {
            console.error("Speech Synthesis Error:", e);
        }
    }

    if ('speechSynthesis' in window) { window.speechSynthesis.getVoices(); }

    function triggerCamera() {
        const fileInput = document.createElement('input');
        fileInput.type = 'file';
        fileInput.accept = 'image/*';
        fileInput.capture = 'environment';
        
        fileInput.onchange = (event) => handleUpload(event);
        fileInput.click();
    }

    async function handleUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        btnCurrency.disabled = true;
        viewfinder.classList.remove('is-result');
        viewfinder.classList.add('is-processing');
        
        statusEl.innerHTML = 'Uploading to ML Engine... <span class="blink">_</span>';
        resultText.innerHTML = 'Analyzing&hellip;';
        resultSub.textContent = 'Processing visual features';
        vfArt.innerHTML = '<span class="glyph">...</span>';

        const formData = new FormData();
        formData.append('image', file);

        try {
            const response = await fetch(`/scan`, {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();

            if (response.ok) {
                viewfinder.classList.remove('is-processing');
                viewfinder.classList.add('is-result');
                statusEl.textContent = 'Done.';

                vfArt.innerHTML = `<span class="glyph">₹${data.value || '?'}</span>`;
                resultText.textContent = data.value ? `₹${data.value}` : "Analysis Failed";
                resultSub.textContent = `Match confidence: ${(data.confidence * 100).toFixed(1)}%`;

                speakAloud(data.spoken);
            } else {
                throw new Error(data.error || "Server failed to process image");
            }
        } catch (error) {
            viewfinder.classList.remove('is-processing');
            statusEl.textContent = 'Error.';
            resultText.textContent = "Processing Failed";
            resultSub.textContent = error.message;
            vfArt.innerHTML = `<span class="glyph">X</span>`;
            speakAloud("Sorry, the scan failed. Please try again.");
            console.error('API Error:', error);
        } finally {
            btnCurrency.disabled = false;
        }
    }

    formCurrency.addEventListener('submit', function (e) {
        e.preventDefault();
        triggerCamera();
    });
})();