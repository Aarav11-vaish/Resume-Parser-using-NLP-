document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('uploadForm');
    form.addEventListener('submit', function(event) {
        event.preventDefault();
        const formData = new FormData(form);

        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                displayResults(data);
            } else {
                alert('Error: ' + data.message);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Error uploading file.');
        });
    });

    function displayResults(data) {
        document.getElementById('phone_number').textContent = data.phone_number || 'Not found';
        document.getElementById('emails').textContent = data.emails.join(', ') || 'Not found';
        document.getElementById('urls').textContent = data.urls.join(', ') || 'Not found';
        document.getElementById('hyperlinks').textContent = data.hyperlinks.join(', ') || 'Not found';
        document.getElementById('skills').textContent = data.skills.join(', ') || 'Not found';
        document.getElementById('education').textContent = data.education.join(', ') || 'Not found';
        document.getElementById('summary').textContent = data.summary || 'Not found';
        document.getElementById('results').classList.remove('hidden');
    }
});
