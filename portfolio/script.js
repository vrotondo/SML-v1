// Typing animation
const phrases = [
  'ML Engineer',
  'AI Systems Builder',
  'LLM Enthusiast',
  'Agent Architect',
];
let phraseIdx = 0;
let charIdx = 0;
let deleting = false;
const typingEl = document.getElementById('typing');

function type() {
  const current = phrases[phraseIdx];
  if (deleting) {
    typingEl.textContent = current.slice(0, --charIdx);
    if (charIdx === 0) {
      deleting = false;
      phraseIdx = (phraseIdx + 1) % phrases.length;
      setTimeout(type, 500);
      return;
    }
  } else {
    typingEl.textContent = current.slice(0, ++charIdx);
    if (charIdx === current.length) {
      deleting = true;
      setTimeout(type, 1800);
      return;
    }
  }
  setTimeout(type, deleting ? 60 : 90);
}
type();

// Nav scroll effect
const nav = document.getElementById('nav');
window.addEventListener('scroll', () => {
  nav.classList.toggle('scrolled', window.scrollY > 40);
}, { passive: true });

// Mobile hamburger
const hamburger = document.getElementById('hamburger');
const navMobile = document.getElementById('nav-mobile');
hamburger.addEventListener('click', () => {
  navMobile.classList.toggle('open');
});
navMobile.querySelectorAll('a').forEach(a =>
  a.addEventListener('click', () => navMobile.classList.remove('open'))
);

// Scroll reveal
const observer = new IntersectionObserver(
  entries => entries.forEach(e => {
    if (e.isIntersecting) {
      e.target.classList.add('visible');
      observer.unobserve(e.target);
    }
  }),
  { threshold: 0.1 }
);
document.querySelectorAll('.project-card, .about-card, .skill-group, .contact-card, .about-text, .about-stats')
  .forEach(el => {
    el.classList.add('reveal');
    observer.observe(el);
  });

// Expand/collapse toggle
function toggleExpand(contentId, wrapperId) {
  const content = document.getElementById(contentId);
  const wrapper = document.getElementById(wrapperId);
  const btn = wrapper.querySelector('.expand-btn');
  const isHidden = content.hidden;
  content.hidden = !isHidden;
  btn.textContent = isHidden
    ? btn.textContent.replace('▾', '▴').replace('Show', 'Hide')
    : btn.textContent.replace('▴', '▾').replace('Hide', 'Show');
}
