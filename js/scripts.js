document.addEventListener("DOMContentLoaded", function () {
    const generateButton = document.getElementById("generateButton");
    const inputText = document.getElementById("inputText");
    const captionResult = document.getElementById("captionResult");

    generateButton.addEventListener("click", function () {
        const text = inputText.value;
        fetch("/generate_caption", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text: text })
        })
        .then(response => response.json())
        .then(data => {
            captionResult.textContent = data.caption;
        })
        .catch(error => {
            console.error("Error:", error);
        });
    });
});