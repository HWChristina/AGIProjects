document.getElementById('uploadForm').addEventListener('submit', async function(event) {
    event.preventDefault();

    const fileInput = document.getElementById('pdfFile');
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    const response = await fetch('/upload', {
        method: 'POST',
        body: formData
    });

    const result = await response.json();
    document.getElementById('uploadMessage').innerText = result.message || 'Upload failed';
});

document.getElementById('askForm').addEventListener('submit', async function(event) {
    event.preventDefault();

    const question = document.getElementById('question').value;
    const response = await fetch('/ask', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ question: question })
    });

    const result = await response.json();
    document.getElementById('answer').innerText = result.answer || 'Failed to get answer';
});
