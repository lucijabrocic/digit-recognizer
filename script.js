const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clearBtn');
const predictBtn = document.getElementById('predictBtn');
const resultDiv = document.getElementById('result');
const predictionSpan = document.getElementById('prediction');
const confidenceSpan = document.getElementById('confidence');
const probabilitiesDiv = document.getElementById('probabilities');

let isDrawing = false;
let model = null;

// Postavi canvas za crtanje
ctx.fillStyle = 'white';
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.strokeStyle = 'black';
ctx.lineWidth = 15;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';

// Učitaj MNIST model
async function loadModel() {
    try {
        // Koristi pre-trained MNIST model
        model = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mnist_v1/model.json');
        console.log('Model učitan uspješno!');
        predictBtn.disabled = false;
    } catch (error) {
        console.error('Greška pri učitavanju modela:', error);
        alert('Greška pri učitavanju AI modela. Molimo osvježite stranicu.');
    }
}

// Funkcije za crtanje - desktop (miš)
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

// Funkcije za crtanje - mobitel (dodir)
canvas.addEventListener('touchstart', handleTouchStart);
canvas.addEventListener('touchmove', handleTouchMove);
canvas.addEventListener('touchend', stopDrawing);

function startDrawing(e) {
    isDrawing = true;
    const rect = canvas.getBoundingClientRect();
    ctx.beginPath();
    ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
}

function draw(e) {
    if (!isDrawing) return;
    const rect = canvas.getBoundingClientRect();
    ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
    ctx.stroke();
}

function handleTouchStart(e) {
    e.preventDefault();
    isDrawing = true;
    const rect = canvas.getBoundingClientRect();
    const touch = e.touches[0];
    ctx.beginPath();
    ctx.moveTo(touch.clientX - rect.left, touch.clientY - rect.top);
}

function handleTouchMove(e) {
    e.preventDefault();
    if (!isDrawing) return;
    const rect = canvas.getBoundingClientRect();
    const touch = e.touches[0];
    ctx.lineTo(touch.clientX - rect.left, touch.clientY - rect.top);
    ctx.stroke();
}

function stopDrawing() {
    isDrawing = false;
}

// Obriši canvas
clearBtn.addEventListener('click', () => {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    resultDiv.classList.add('hidden');
    probabilitiesDiv.classList.add('hidden');
});

// Predviđanje
predictBtn.addEventListener('click', async () => {
    if (!model) {
        alert('Model se još učitava...');
        return;
    }
    
    // Pretvori canvas u 28x28 sliku (MNIST format)
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const tensor = preprocessImage(imageData);
    
    // Predviđanje
    const prediction = await model.predict(tensor);
    const probabilities = await prediction.data();
    
    // Pronađi znamenku s najvećom vjerojatnosti
    const predictedDigit = probabilities.indexOf(Math.max(...probabilities));
    const confidence = (Math.max(...probabilities) * 100).toFixed(2);
    
    // Prikaži rezultat
    predictionSpan.textContent = predictedDigit;
    confidenceSpan.textContent = confidence;
    resultDiv.classList.remove('hidden');
    
    // Prikaži sve vjerojatnosti
    displayProbabilities(probabilities);
    
    // Očisti memoriju
    tensor.dispose();
    prediction.dispose();
});

function preprocessImage(imageData) {
    return tf.tidy(() => {
        // Pretvori u tensor
        let tensor = tf.browser.fromPixels(imageData, 1);
        
        // Skaliraj na 28x28
        tensor = tf.image.resizeBilinear(tensor, [28, 28]);
        
        // Invertiraj boje (MNIST ima bijele znamenke na crnoj pozadini)
        tensor = tf.scalar(255).sub(tensor);
        
        // Normaliziraj na [0, 1]
        tensor = tensor.div(tf.scalar(255));
        
        // Reshape za model [1, 28, 28, 1]
        tensor = tensor.expandDims(0);
        
        return tensor;
    });
}

function displayProbabilities(probabilities) {
    probabilitiesDiv.innerHTML = '<h3 style="margin-bottom: 15px; color: #333;">Vjerojatnosti:</h3>';
    
    // Sortiraj znamenke po vjerojatnosti
    const sortedProbs = Array.from(probabilities)
        .map((prob, digit) => ({ digit, prob }))
        .sort((a, b) => b.prob - a.prob);
    
    // Prikaži top 5
    sortedProbs.slice(0, 5).forEach(({ digit, prob }) => {
        const percentage = (prob * 100).toFixed(1);
        const barHTML = `
            <div class="prob-bar">
                <div class="prob-label">
                    <span>Znamenka ${digit}</span>
                    <span>${percentage}%</span>
                </div>
                <div class="prob-fill-container">
                    <div class="prob-fill" style="width: ${percentage}%"></div>
                </div>
            </div>
        `;
        probabilitiesDiv.innerHTML += barHTML;
    });
    
    probabilitiesDiv.classList.remove('hidden');
}

// Učitaj model pri učitavanju stranice
loadModel();
