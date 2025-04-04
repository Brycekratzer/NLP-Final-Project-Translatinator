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

        // Clear the transcription text
        transcriptionElement.innerText = "";

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

                // Refresh history
                await getEntries();

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
stopBtn.addEventListener("click", async () => {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
    }
    startBtn.disabled = false;
    stopBtn.disabled = true;

    // DEPRECATED
    // await sleep(2000).then(async () => {
    //     console.log("sleep done!");
    //     // Update history
    //     await getEntries();
    // });

});

// Translate text
translateBtn.addEventListener("click", async () => {

    // Disable all other buttons during translation process
    startBtn.disabled = true;
    pronounceBtn.disabled = true;

    // Response is no longer const as it must account for 2 states
    let response;

    if (transcriptionElement.textContent.length != 0) {
        // There is a transcription (selected)

        // Create a text from transcription
        const textBlob = new Blob([transcriptionElement.textContent], { type: "text/plain" });
        const filename = "transcription.txt";

        // Prepare FormData
        const formData = new FormData();
        formData.append("text", textBlob, filename);

        // Send to Flask backend (with body)
        response = await fetch("/translateselected", {
            method: "POST",
            body: formData
        });
    } else {

        // Send to Flask backend
        response = await fetch("/translate", {
            method: "POST"
        });
    }


    if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
    }

    const data = await response.json();

    // Display translation result
    translationElement.textContent = data["translation"] || "Error translating audio.";

    // Enable buttons after translation
    startBtn.disabled = false;
    pronounceBtn.disabled = false;

});

// Kinda like a sleep function! [NOT USED CURRENTLY]
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function getCode(str, char1, char2) {
    const string = String(str);
    const underscoreLocation = string.indexOf(char1);
    const dotLocation = string.indexOf(char2);

    return string.substring(underscoreLocation, dotLocation);
}

refreshBtn.addEventListener("click", async () => {
    await getEntries();
    console.log('Refreshing!');
});

const clickHandler = (num) => {
    console.log(`Clicked button ${num}!`);
};

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
    }

    // Grab zipped file and treat as blob
    let data = response.blob();

    // Unzipping the file
    zip.loadAsync(data).then((zip) => {
        const files = zip.files;
        const reverseFiles = Array.from(Object.entries(files)).reverse();
        // console.log(reverseFiles);

        // console.log(Object.keys(files).length);

        // Number of entries
        // const numOfPairs = Object.keys(files).length / 3;
        const dictionary = new Map();

        // Keep track of what button is clicked
        let i = 0;

        // Go through all the keys(files) in the unzipped directory
        Object.keys(files).forEach((filename) => {

            // Grab the code of the file
            const code = getCode(filename, "_", ".");

            // Grabbing a file
            const element = files[filename];

            if (element.name.endsWith(".txt")) {
                // Handle as a text file

                element.async('text').then((data) => {
                    if (!dictionary.get(code)) {
                        // Add new entry
                        dictionary.set(code, String(data));
                    }
                })

            }


            if (element.name.endsWith(".webm")) {

                // Extract as blob
                element.async('blob').then((data) => {
                    // console.log(data);

                    // Create the element to be added to the list
                    const listitem = document.createElement("li");

                    // Create button so it can be used to link the file when it needs to be sent to the main translation part
                    const button = document.createElement("button");
                    button.innerText = "Select";

                    const transcription = dictionary.get(code);
                    if (!transcription) {
                        throw new Error("Missing transcription file! History is cooked!");
                    }



                    // button.onclick(clickHandler(i++));
                    button.onclick = () => {
                        transcriptionElement.innerText = transcription;
                    }
                    // button.addEventListener("click", clickHandler(i++));

                    // button.
                    // button.classList("history-button");

                    // Build audio controls
                    const audioURL = URL.createObjectURL(data);
                    const figure = document.createElement("figure");
                    const caption = document.createElement("figcaption");
                    const audio = document.createElement("audio");

                    // Set name
                    // caption.innerText = element.name;
                    caption.innerText = transcription;

                    // Enable controls
                    audio.controls = true;
                    audio.style.paddingTop = "5px";

                    // Set source to the audio file
                    audio.src = audioURL;

                    // Text for transcription
                    // const text = document.createElement("p");
                    // text.innerText = transcription;

                    // Building the the element
                    figure.appendChild(caption);
                    figure.appendChild(audio);
                    figure.appendChild(button);
                    // figure.appendChild(text);


                    // Add the whole element to the list
                    listitem.appendChild(figure);
                    historylist.appendChild(listitem);

                    // Something went wrong when blobbing the data
                }).catch((error) => {
                    console.error('An error occured when trying to "blob" the file data!');
                });

                // console.log('Handling .webm!');
            }
        });

        // Catch any errors
    }).catch((error) => {
        console.error("There was a problem unzipping the file");
    });

});

await getEntries();


