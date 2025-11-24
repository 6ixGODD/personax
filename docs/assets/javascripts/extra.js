// Add copy button to code blocks
document.addEventListener('DOMContentLoaded', function () {
    // Add IDs to headings for better navigation
    document.querySelectorAll('h2, h3, h4').forEach(function (heading) {
        if (!heading.id) {
            heading.id = heading.textContent
                .toLowerCase()
                .replace(/[^\w\s-]/g, '')
                .replace(/\s+/g, '-');
        }
    });

    // Smooth scroll for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Add external link indicators
    document.querySelectorAll('a[href^="http"]').forEach(link => {
        if (!link.hostname.includes(window.location.hostname)) {
            link.setAttribute('target', '_blank');
            link.setAttribute('rel', 'noopener noreferrer');
        }
    });
});

// Version warning for development docs
if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    console.log('📚 Viewing BoxLab documentation locally');
}
