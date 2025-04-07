let startBtn = document.getElementById("startBtn");
let stopBtn = document.getElementById("stopBtn");
let transcriptionElement = document.getElementById("transcription");
let translationElement = document.getElementById("translatedText");
let en_spa_Btn = document.getElementById("en_spa");
let spa_en_Btn = document.getElementById("spa_en");
let pronounceBtn = document.getElementById("pronouncetxt");

let mediaRecorder;
let audioChunks = [];

// Start recording
startBtn.addEventListener("click", async () => {
    try {
        // Clear previous recordings
        audioChunks = [];
        startBtn.disabled = true;
        stopBtn.disabled = false;

        // Get microphone access
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
            try {
                // Create a Blob from recorded audio
                const audioBlob = new Blob(audioChunks, { type: "audio/webm" });
                const filename = "audio.webm";

                // Prepare FormData
                const formData = new FormData();
                formData.append("audio", audioBlob, filename);

                // Send to Flask backend
                const response = await fetch("/transcribe", {
                    method: "POST",
                    body: formData
                });

                if (!response.ok) {
                    throw new Error(`Server error: ${response.status}`);
                }

                const data = await response.json();

                // Display transcription result
                transcriptionElement.textContent = data["error"] || data["transcription"] || "Error transcribing audio.";

                // Enable buttons
                en_spa_Btn.disabled = false;
                spa_en_Btn.disabled = false;
                pronounceBtn.disabled = false;
            } catch (error) {
                transcriptionElement.textContent = "Processing error.";
                console.error("Processing error:", error);
            }
        };

        mediaRecorder.start();
    } catch (error) {
        console.error("Error accessing microphone or starting recorder:", error);
        startBtn.disabled = false;
        stopBtn.disabled = true;
    }
});

// Stop recording
stopBtn.addEventListener("click", () => {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
    }
    startBtn.disabled = false;
    stopBtn.disabled = true;
});

// Translate english to spanish text
en_spa_Btn.addEventListener("click", async () => {

    // Disable all other buttons during translation process
    startBtn.disabled = true;
    pronounceBtn.disabled = true;
    spa_en_Btn.disabled = true;

    // Send to Flask backend
    const response = await fetch("/en_spa_trans", {
        method: "POST"
    });

    if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
    }

    const data = await response.json();

    // Display translation result
    translationElement.textContent = data["translation"] || "Error translating audio.";

    // Enable buttons after translation
    startBtn.disabled = false;
    pronounceBtn.disabled = false;
    spa_en_Btn.disabled = false;
});

// Translate spanish to english text
spa_en_Btn.addEventListener("click", async () => {

    // Disable all other buttons during translation process
    startBtn.disabled = true;
    pronounceBtn.disabled = true;
    spa_en_Btn.disabled = true;

    // Send to Flask backend
    const response = await fetch("/spa_en_trans", {
        method: "POST"
    });

    if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
    }

    const data = await response.json();

    // Display translation result
    translationElement.textContent = data["translation"] || "Error translating audio.";

    // Enable buttons after translation
    startBtn.disabled = false;
    pronounceBtn.disabled = false;
    spa_en_Btn.disabled = false;
});


