let startBtn = document.getElementById("startBtn");
let stopBtn = document.getElementById("stopBtn");
let transcriptionElement = document.getElementById("transcription");
let translationElement = document.getElementById("translatedText");
let translateBtn = document.getElementById("translatetxt");
let pronounceBtn = document.getElementById("pronouncetxt");

const refreshBtn = document.querySelector("button");
const historylist = document.querySelector("ol");

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
                translateBtn.disabled = false;
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

// Translate text
translateBtn.addEventListener("click", async () => {

    // Disable all other buttons during translation process
    startBtn.disabled = true;
    pronounceBtn.disabled = true;

    // Send to Flask backend
    const response = await fetch("/translate", {
        method: "POST"
    });

    if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
    }

    const data = await response.json();

    // Update history
    getEntries();

    // Display translation result
    translationElement.textContent = data["translation"] || "Error translating audio.";

    // Enable buttons after translation
    startBtn.disabled = false;
    pronounceBtn.disabled = false;
});

function getCode(str, char1, char2) {
    
}

refreshBtn.addEventListener("click", async () => {
    getEntries();
    console.log('Refreshing!');
});

const getEntries = (async () => {

    // Clear list
    historylist.innerHTML = "";

    const zip = new JSZip();
    
    // Grabbing the entries in the uploads folder
    const response = await fetch("/collect", {
        method: "GET"
    });
    
    if (!response.ok) {
        throw new Error(`Something went wrong! ERROR: ${response.status}`);
    } else {
        let data = response.blob();
        JSZip.loadAsync(data).then((zip) => {
            const files = zip.files;
        
            console.log(Object.keys(files).length);
            // Number of entries
            const numOfPairs = Object.keys(files).length / 3;
            const dictionary = {};

            Object.keys(files).forEach((filename) => {
                console.log(files[filename]);
                if (filename.name.endsWith(".webm")) {

                    // const audioURL = URL.createObjectURL(filename);

                    let listitem = document.createElement("li");
                    let audio = document.createElement("audio");

                    listitem.appendChild(audio);
                    historylist.appendChild(listitem);

                    console.log('Handling .webm!');
                }
            });

        });
    }
    
});

getEntries();


