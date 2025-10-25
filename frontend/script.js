document.getElementById("translateBtn").addEventListener("click", async () => {
    const inputText = document.getElementById("inputText").value.trim();
    const outputBox = document.getElementById("outputText");

    if (!inputText) {
        outputBox.textContent = "⚠️ Please enter French text first.";
        return;
    }

    outputBox.textContent = "⏳ Translating...";

    try {
        const response = await fetch("/translate/", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text: inputText })
        });

        if (!response.ok) {
            throw new Error("Translation API error.");
        }

        const data = await response.json();
        outputBox.textContent = data.predicted_english;
    } catch (error) {
        console.error(error);
        outputBox.textContent = "❌ Failed to translate. Please try again.";
    }
});
