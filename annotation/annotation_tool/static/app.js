document.addEventListener('DOMContentLoaded', () => {
    const loginBtn = document.getElementById('login-btn');
    const logoutBtn = document.getElementById('logout-btn');
    const agreeBtn = document.getElementById('agree-btn');
    const disagreeBtn = document.getElementById('disagree-btn');
    const prevBtn = document.getElementById('prev-btn');
    const nextBtn = document.getElementById('next-btn');
    const clearBtn = document.getElementById('clear-btn');
    const indexInput = document.getElementById('index-input');
    
    if (loginBtn) {
        loginBtn.addEventListener('click', login);
    }
    
    if (logoutBtn) {
        logoutBtn.addEventListener('click', logout);
        loadQuestion();
    }
    
    if (agreeBtn) {
        agreeBtn.addEventListener('click', () => submitAgreement(true));
    }
    
    if (disagreeBtn) {
        disagreeBtn.addEventListener('click', () => submitAgreement(false));
    }
    
    if (prevBtn) {
        prevBtn.addEventListener('click', () => navigate(-1));
    }
    
    if (nextBtn) {
        nextBtn.addEventListener('click', () => navigate(1));
    }
    
    if (clearBtn) {
        clearBtn.addEventListener('click', clearResponse);
    }

    if (indexInput) {
        indexInput.addEventListener('change', () => {
            let val = parseInt(indexInput.value);
            if (isNaN(val)) val = 1;
            // Clamp value
            // We don't know totalQuestions here easily unless we store it globally, which we do.
            if (val < 1) val = 1;
            if (totalQuestions > 0 && val > totalQuestions) val = totalQuestions;
            
            loadQuestion(val - 1); // 0-indexed
        });
        
        // Also handle Enter key
        indexInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                indexInput.blur(); // Triggers change
            }
        });
    }
});

async function login() {
    const username = document.getElementById('username-input').value.trim();
    if (!username) return alert('Please enter a username');
    
    try {
        const res = await fetch('/login', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({username})
        });
        
        if (res.ok) {
            window.location.reload();
        } else {
            alert('Login failed');
        }
    } catch (e) {
        console.error(e);
        alert('Error logging in');
    }
}

async function logout() {
    try {
        await fetch('/logout', {method: 'POST'});
        window.location.reload();
    } catch (e) {
        console.error(e);
    }
}

let currentQuestionIndex = null;
let currentSelectedChoice = null;
let totalQuestions = 0;

async function loadQuestion(index = null) {
    const container = document.getElementById('question-container');
    const loading = document.getElementById('loading');
    const finished = document.getElementById('finished');
    const judgeCard = document.getElementById('judge-card');
    const optionsCard = document.getElementById('options-card');
    const clearBtn = document.getElementById('clear-btn');
    const indexInput = document.getElementById('index-input');
    
    container.classList.add('hidden');
    finished.classList.add('hidden');
    judgeCard.classList.add('hidden');
    optionsCard.classList.remove('hidden');
    loading.classList.remove('hidden');
    clearBtn.classList.add('hidden');
    
    let url = '/get_question';
    if (index !== null) {
        url += `?index=${index}`;
    }
    
    try {
        const res = await fetch(url);
        const data = await res.json();
        
        loading.classList.add('hidden');
        
        if (data.message === 'All questions answered!' && index === null) {
            finished.classList.remove('hidden');
            return;
        }
        
        if (data.error) {
            alert(data.error);
            return;
        }
        
        currentQuestionIndex = data.index;
        totalQuestions = data.total;
        currentSelectedChoice = null;
        
        // Update Nav
        if (indexInput) {
            indexInput.value = currentQuestionIndex + 1;
            indexInput.max = totalQuestions;
        }
        document.getElementById('total-count').textContent = totalQuestions;
        document.getElementById('prev-btn').disabled = currentQuestionIndex === 0;
        document.getElementById('next-btn').disabled = currentQuestionIndex === totalQuestions - 1;
        
        // Populate UI
        document.getElementById('dimension-name').textContent = data.dimension_name;
        document.getElementById('dimension-description').textContent = data.dimension_description;
        document.getElementById('why-differ').textContent = data.why_differ;
        document.getElementById('how-subtle').textContent = data.how_subtle;
        document.getElementById('question-text').textContent = data.question;
        document.getElementById('personalized-response').textContent = data.personalized_response;
        
        // Render Options
        const optionsContainer = document.getElementById('options-container');
        optionsContainer.innerHTML = '';
        
        const userAnnotation = data.user_annotation;
        
        data.options.forEach((opt, idx) => {
            const letter = String.fromCharCode(65 + idx); // A, B, C...
            const el = document.createElement('div');
            el.className = 'option-item';
            el.innerHTML = `<span class="option-letter">${letter}</span><span>${opt}</span>`;
            
            if (userAnnotation) {
                el.style.pointerEvents = 'none';
                if (userAnnotation.user_choice === letter) {
                    el.classList.add('selected');
                    if (userAnnotation.is_correct) {
                        el.style.borderColor = 'var(--success)';
                    } else {
                        el.style.borderColor = 'var(--error)';
                    }
                }
            } else {
                el.onclick = () => submitChoice(letter);
            }
            optionsContainer.appendChild(el);
        });
        
        if (userAnnotation) {
            clearBtn.classList.remove('hidden');
            if (!userAnnotation.is_correct) {
                // Show judge card if they got it wrong and agreed/disagreed
                // Actually, if they already answered, we might not show the judge card unless we want to show history.
                // For simplicity, just show "Answered: [Result]" or similar.
                // But the user might want to see what they did.
                // If incorrect, we could show the judge rationale again?
                // Let's keep it simple: just show selection state.
            }
        }
        
        container.classList.remove('hidden');
        
    } catch (e) {
        console.error(e);
        loading.textContent = 'Error loading question.';
    }
}

function navigate(direction) {
    if (currentQuestionIndex === null) return;
    const newIndex = currentQuestionIndex + direction;
    if (newIndex >= 0 && newIndex < totalQuestions) {
        loadQuestion(newIndex);
    }
}

async function clearResponse() {
    if (!confirm('Are you sure you want to clear your response?')) return;
    
    try {
        const res = await fetch('/clear_response', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({index: currentQuestionIndex})
        });
        
        if (res.ok) {
            loadQuestion(currentQuestionIndex);
        } else {
            alert('Failed to clear response');
        }
    } catch (e) {
        console.error(e);
        alert('Error clearing response');
    }
}

async function submitChoice(choice) {
    currentSelectedChoice = choice;
    
    // Disable options to prevent double click
    const options = document.querySelectorAll('.option-item');
    options.forEach(opt => opt.style.pointerEvents = 'none');
    
    try {
        const res = await fetch('/submit_choice', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                index: currentQuestionIndex,
                choice: choice
            })
        });
        
        const data = await res.json();
        
        if (data.correct) {
            // Correct! Load next
            loadQuestion(currentQuestionIndex + 1);
        } else {
            // Incorrect. Show judge rationale.
            document.getElementById('options-card').classList.add('hidden');
            const judgeCard = document.getElementById('judge-card');
            judgeCard.classList.remove('hidden');
            document.getElementById('judge-rationale').textContent = data.judge_rationale;
            document.getElementById('correct-answer-display').textContent = data.correct_choice;
        }
        
    } catch (e) {
        console.error(e);
        alert('Error submitting choice');
        // Re-enable options
        options.forEach(opt => opt.style.pointerEvents = 'auto');
    }
}

async function submitAgreement(agree) {
    const btn = agree ? document.getElementById('agree-btn') : document.getElementById('disagree-btn');
    btn.disabled = true;
    
    try {
        const res = await fetch('/submit_agreement', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                index: currentQuestionIndex,
                choice: currentSelectedChoice,
                agree: agree
            })
        });
        
        if (res.ok) {
            loadQuestion(currentQuestionIndex + 1);
        } else {
            alert('Failed to submit agreement');
        }
    } catch (e) {
        console.error(e);
        alert('Error submitting agreement');
    } finally {
        btn.disabled = false;
    }
}
