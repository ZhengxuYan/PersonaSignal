document.addEventListener('DOMContentLoaded', () => {
    const loginBtn = document.getElementById('login-btn');
    const logoutBtn = document.getElementById('logout-btn');
    const yesBtn = document.getElementById('yes-btn');
    const noBtn = document.getElementById('no-btn');
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
    
    if (yesBtn) {
        yesBtn.addEventListener('click', () => submitVerification(true));
    }
    
    if (noBtn) {
        noBtn.addEventListener('click', () => submitVerification(false));
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
            if (val < 1) val = 1;
            if (totalQuestions > 0 && val > totalQuestions) val = totalQuestions;
            loadQuestion(val - 1);
        });
        indexInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') indexInput.blur();
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
        if (res.ok) window.location.reload();
        else alert('Login failed');
    } catch (e) {
        console.error(e);
        alert('Error logging in');
    }
}

async function logout() {
    try {
        await fetch('/logout', {method: 'POST'});
        window.location.reload();
    } catch (e) { console.error(e); }
}

let currentQuestionIndex = null;
let totalQuestions = 0;

async function loadQuestion(index = null) {
    const container = document.getElementById('question-container');
    const loading = document.getElementById('loading');
    const finished = document.getElementById('finished');
    const clearBtn = document.getElementById('clear-btn');
    const indexInput = document.getElementById('index-input');
    const yesBtn = document.getElementById('yes-btn');
    const noBtn = document.getElementById('no-btn');
    const statusMsg = document.getElementById('status-message');
    
    container.classList.add('hidden');
    finished.classList.add('hidden');
    loading.classList.remove('hidden');
    clearBtn.classList.add('hidden');
    statusMsg.classList.add('hidden');
    yesBtn.disabled = false;
    noBtn.disabled = false;
    yesBtn.classList.remove('selected-btn');
    noBtn.classList.remove('selected-btn');
    
    let url = '/get_question';
    if (index !== null) url += `?index=${index}`;
    
    try {
        const res = await fetch(url);
        const data = await res.json();
        
        loading.classList.add('hidden');
        
        if (data.error) {
            alert(data.error);
            return;
        }
        
        currentQuestionIndex = data.index;
        totalQuestions = data.total;
        
        if (indexInput) {
            indexInput.value = currentQuestionIndex + 1;
            indexInput.max = totalQuestions;
        }
        document.getElementById('total-count').textContent = totalQuestions;
        document.getElementById('prev-btn').disabled = currentQuestionIndex === 0;
        document.getElementById('next-btn').disabled = currentQuestionIndex === totalQuestions - 1;
        
        document.getElementById('dimension-name').textContent = data.dimension_name;
        document.getElementById('ground-truth-persona').textContent = data.ground_truth_persona;
        document.getElementById('question-text').textContent = data.question;
        document.getElementById('personalized-response').textContent = data.personalized_response;
        
        if (data.user_annotation) {
            clearBtn.classList.remove('hidden');
            yesBtn.disabled = true;
            noBtn.disabled = true;
            
            if (data.user_annotation.is_personalized) {
                yesBtn.classList.add('selected-btn');
                statusMsg.textContent = "You selected: Yes";
            } else {
                noBtn.classList.add('selected-btn');
                statusMsg.textContent = "You selected: No";
            }
            statusMsg.classList.remove('hidden');
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
        if (res.ok) loadQuestion(currentQuestionIndex);
        else alert('Failed to clear response');
    } catch (e) {
        console.error(e);
        alert('Error clearing response');
    }
}

async function submitVerification(isPersonalized) {
    try {
        const res = await fetch('/submit_verification', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                index: currentQuestionIndex,
                is_personalized: isPersonalized
            })
        });
        
        if (res.ok) {
            loadQuestion(currentQuestionIndex + 1);
        } else {
            alert('Failed to submit');
        }
    } catch (e) {
        console.error(e);
        alert('Error submitting');
    }
}
