import './style.css'

const fileInput = document.getElementById('file-upload')
const textInput = document.getElementById('text-input')
const runBtn = document.getElementById('run-btn')
const imagePreview = document.getElementById('image-preview')
const resultsPanel = document.getElementById('results-panel')

const resClass = document.getElementById('res-class')
const resConf = document.getElementById('res-conf')
const resBinary = document.getElementById('res-binary')
const resOcr = document.getElementById('res-ocr')
const resModels = document.getElementById('res-models')

// Nodes
const nodes = [
  document.getElementById('node-input'),
  document.getElementById('node-img-enc'),
  document.getElementById('node-txt-enc'),
  document.getElementById('node-fusion'),
  document.getElementById('node-clf'),
  document.getElementById('node-output')
]

let selectedFile = null

// Handle file selection
fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0]
  if (file) {
    selectedFile = file
    const reader = new FileReader()
    reader.onload = (e) => {
      imagePreview.src = e.target.result
      imagePreview.hidden = false
    }
    reader.readAsDataURL(file)
    runBtn.disabled = false
  }
})

// Animate nodes in sequence
const animatePipeline = async () => {
  // Reset all
  nodes.forEach(n => n.classList.remove('active'))
  resultsPanel.classList.add('hidden')
  
  const delay = ms => new Promise(res => setTimeout(res, ms))

  // 1. Input
  nodes[0].classList.add('active')
  await delay(600)
  nodes[0].classList.remove('active')

  // 2. Encoders (Parallel)
  nodes[1].classList.add('active')
  nodes[2].classList.add('active')
  await delay(800)
  nodes[1].classList.remove('active')
  nodes[2].classList.remove('active')

  // 3. Fusion
  nodes[3].classList.add('active')
  await delay(700)
  nodes[3].classList.remove('active')

  // 4. Classifier
  nodes[4].classList.add('active')
  await delay(600)
  nodes[4].classList.remove('active')

  // 5. Output
  nodes[5].classList.add('active')
}

// Handle run button click
runBtn.addEventListener('click', async () => {
  if (!selectedFile) return
  
  runBtn.disabled = true
  runBtn.innerHTML = 'Processing...'
  
  // Start animation but don't await immediately
  const animPromise = animatePipeline()

  try {
    const formData = new FormData()
    formData.append('image', selectedFile)
    if (textInput.value.trim() !== '') {
      formData.append('text', textInput.value)
    }

    const response = await fetch('http://localhost:8000/predict', {
      method: 'POST',
      body: formData
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    const data = await response.json()
    
    // Wait for animation to finish before showing results
    await animPromise
    
    // Display results
    resClass.textContent = data.primary_class
    resConf.textContent = (data.confidence * 100).toFixed(1) + '%'
    resBinary.textContent = data.binary_result
    resOcr.textContent = data.ocr_text || 'None'
    resModels.textContent = data.models_used.join(', ')
    
    // Change color based on class
    const colors = {
      'NotHate': '#10b981',
      'Racist': '#ef4444',
      'Sexist': '#f59e0b',
      'Homophobe': '#8b5cf6',
      'Religion': '#0ea5e9',
      'OtherHate': '#f43f5e'
    }
    resClass.style.color = colors[data.primary_class] || '#fff'
    resBinary.style.color = data.binary_result === 'Hateful' ? '#ef4444' : '#10b981'

    resultsPanel.classList.remove('hidden')
    
  } catch (error) {
    console.error('Prediction failed:', error)
    await animPromise
    resClass.textContent = 'Error'
    resConf.textContent = '0.0%'
    resBinary.textContent = '-'
    resClass.style.color = 'red'
    resultsPanel.classList.remove('hidden')
  } finally {
    runBtn.disabled = false
    runBtn.innerHTML = `
      <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><polygon points="10 8 16 12 10 16 10 8"></polygon></svg>
      Run Pipeline
    `
    setTimeout(() => {
      nodes[5].classList.remove('active')
    }, 2000)
  }
})
