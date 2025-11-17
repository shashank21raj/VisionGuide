// App.jsx - CLEAN VERSION
import { useState, useRef, useEffect } from 'react'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { 
  faPlay, 
  faStop, 
  faVideo, 
  faVideoSlash, 
  faSync, 
  faPause, 
  faMicrophone, 
  faMicrophoneSlash 
} from '@fortawesome/free-solid-svg-icons'
import CanvasBackground from './CanvasBackground'
import './App.css'

function App() {
  const [cameraActive, setCameraActive] = useState(false)
  const [fetchedText, setFetchedText] = useState('Starting up...')
  const [isSpeaking, setIsSpeaking] = useState(false)
  const [autoMode, setAutoMode] = useState(true)
  const [voices, setVoices] = useState([])
  const [selectedVoice, setSelectedVoice] = useState(null)

  const videoRef = useRef(null)
  const streamRef = useRef(null)
  const wsRef = useRef(null)
  const sendIntervalRef = useRef(null)
  const canvasRef = useRef(null)
  const lastSpokenRef = useRef('')
  const reconnectRef = useRef({ tries: 0 })
  const [connectionState, setConnectionState] = useState('closed')

  const SEND_MS = 150  // 10 FPS for better performance

  // Load voices
  useEffect(() => {
    const loadVoices = () => {
      const v = window.speechSynthesis.getVoices()
      setVoices(v)
      if (v.length > 0) {
        const english = v.find(x => x.lang && x.lang.includes('en')) || v[0]
        setSelectedVoice(english)
      }
    }
    loadVoices()
    window.speechSynthesis.onvoiceschanged = loadVoices
    return () => { window.speechSynthesis.onvoiceschanged = null }
  }, [])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopEverything()
    }
  }, [])

  // Speak logic
  useEffect(() => {
    if (!autoMode) return
    if (!fetchedText) return
    if (!selectedVoice) return
    if (fetchedText === lastSpokenRef.current) return
    
    // Don't speak status messages
    if (fetchedText.includes('Server ready') || fetchedText.includes('Starting up')) {
      return
    }

    window.speechSynthesis.cancel()
    const ut = new SpeechSynthesisUtterance(fetchedText)
    ut.voice = selectedVoice
    ut.rate = 1.0
    ut.pitch = 1.0
    ut.volume = 1.0
    ut.onstart = () => setIsSpeaking(true)
    ut.onend = () => setIsSpeaking(false)
    ut.onerror = () => setIsSpeaking(false)
    window.speechSynthesis.speak(ut)
    lastSpokenRef.current = fetchedText
  }, [fetchedText, selectedVoice, autoMode])

  async function startCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          facingMode: 'environment',
          width: { ideal: 480 }, 
          height: { ideal: 360 },
          frameRate: { ideal: 10, max: 15 }
        } 
      })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
      }
      setCameraActive(true)
      return Promise.resolve()
    } catch (err) {
      console.error('Camera error:', err)
      setFetchedText('Camera access denied. Please check permissions.')
      throw err
    }
  }

  function stopCamera() {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => {
        track.stop()
      })
      streamRef.current = null
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    setCameraActive(false)
  }

  function waitForVideoPlaying(timeout = 5000) {
    return new Promise((resolve, reject) => {
      const v = videoRef.current
      if (!v) {
        reject(new Error('Video element not available'))
        return
      }

      if (v.readyState >= 3) {
        resolve()
        return
      }

      const onPlay = () => {
        cleanup()
        resolve()
      }

      const onError = (err) => {
        cleanup()
        reject(new Error('Video playback error: ' + err))
      }

      const cleanup = () => {
        v.removeEventListener('playing', onPlay)
        v.removeEventListener('error', onError)
        clearTimeout(timeoutId)
      }

      v.addEventListener('playing', onPlay, { once: true })
      v.addEventListener('error', onError, { once: true })

      const timeoutId = setTimeout(() => {
        cleanup()
        resolve()
      }, timeout)
    })
  }

  function stopEverything() {
    console.log('Stopping everything...')
    setAutoMode(false)
    stopSendingLoop()
    
    if (wsRef.current) {
      try {
        wsRef.current.close(1000, 'Normal closure')
      } catch (e) {
        console.error('Error closing WebSocket:', e)
      }
      wsRef.current = null
    }
    
    stopCamera()
    window.speechSynthesis.cancel()
    setIsSpeaking(false)
    setConnectionState('closed')
  }

  function startEverything() {
    console.log('Starting everything...')
    setAutoMode(true)
    startCamera()
      .then(() => waitForVideoPlaying())
      .then(() => {
        console.log('Video playing, opening WebSocket...')
        openWebSocket()
      })
      .catch(err => {
        console.error('Startup error:', err)
        setFetchedText('Startup failed: ' + err.message)
      })
  }

  function toggleEverything() {
    if (autoMode && cameraActive) {
      stopEverything()
    } else {
      startEverything()
    }
  }

  function openWebSocket() {
    if (wsRef.current && [WebSocket.OPEN, WebSocket.CONNECTING].includes(wsRef.current.readyState)) {
      console.log('WebSocket already connected or connecting')
      return
    }

    console.log('Opening new WebSocket connection...')
    setConnectionState('connecting')
    
    const ws = new WebSocket('wss://testamentary-gracie-feverous.ngrok-free.dev/ws');
    ws.binaryType = 'arraybuffer'
    wsRef.current = ws

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        console.log('Received:', data)
        
        if (data.type === 'status') {
          setFetchedText(data.msg || 'Server connected')
        } else if (data.type === 'ping') {
          return
        } else if (data.suggestion) {
          setFetchedText(data.suggestion)
        } else if (data.error) {
          setFetchedText('Error: ' + data.error)
        } else {
          setFetchedText(JSON.stringify(data))
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error)
      }
    }

    ws.onopen = () => {
      console.log('✅ WebSocket connected successfully')
      setConnectionState('open')
      reconnectRef.current.tries = 0
      startSendingLoop()
    }

    ws.onerror = (error) => {
      console.error('❌ WebSocket error:', error)
      setConnectionState('error')
      setFetchedText('Connection error')
    }

    ws.onclose = (event) => {
      console.log(`🔌 WebSocket closed: ${event.code} - ${event.reason}`)
      setConnectionState('closed')
      stopSendingLoop()
      wsRef.current = null

      if (autoMode && cameraActive && event.code !== 1000) {
        const tries = (reconnectRef.current.tries || 0) + 1
        reconnectRef.current.tries = tries
        const delay = Math.min(10000, 1000 * Math.min(tries, 5))
        
        console.log(`Reconnecting in ${delay}ms (attempt ${tries})`)
        setTimeout(() => {
          if (autoMode && cameraActive) {
            openWebSocket()
          }
        }, delay)
      }
    }
  }

  function startSendingLoop() {
    if (sendIntervalRef.current) {
      clearInterval(sendIntervalRef.current)
    }
    
    if (!canvasRef.current) {
      const canvas = document.createElement('canvas')
      canvas.width = 480
      canvas.height = 360
      canvasRef.current = canvas
    }

    console.log('Starting frame sending loop...')
    sendIntervalRef.current = setInterval(sendFrameToServer, SEND_MS)
  }

  function stopSendingLoop() {
    if (sendIntervalRef.current) {
      clearInterval(sendIntervalRef.current)
      sendIntervalRef.current = null
    }
  }

  function sendFrameToServer() {
    const ws = wsRef.current
    const video = videoRef.current
    
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      return
    }
    
    if (!video || video.readyState !== 4) {
      return
    }

    const canvas = canvasRef.current
    const context = canvas.getContext('2d')
    
    try {
      context.drawImage(video, 0, 0, canvas.width, canvas.height)
      
      canvas.toBlob((blob) => {
        if (!blob) return
        
        const reader = new FileReader()
        reader.onload = () => {
          if (ws.readyState === WebSocket.OPEN) {
            try {
              ws.send(reader.result)
            } catch (sendError) {
              console.error('Error sending frame:', sendError)
            }
          }
        }
        reader.readAsArrayBuffer(blob)
      }, 'image/jpeg', 0.6)
      
    } catch (error) {
      console.error('Error capturing frame:', error)
    }
  }

  const handleVoiceChange = (e) => {
    const idx = Number(e.target.value)
    if (!isNaN(idx) && voices[idx]) {
      setSelectedVoice(voices[idx])
    }
  }

  return (
    <div className="app">
      <CanvasBackground />
      <div className="content-overlay">
        <nav className="navbar">
          <div className="nav-content">
            <h1 className="nav-title">Vision Guide</h1>
          </div>
        </nav>

        <div className="main-content">
          <div className="camera-container">
            <video 
              ref={videoRef} 
              autoPlay 
              playsInline 
              muted 
              className="camera-video"
            />
            {!cameraActive && (
              <div className="camera-placeholder">
                Camera Off
              </div>
            )}
          </div>

          <div className="controls-center">
            <button 
              onClick={toggleEverything} 
              className={`control-button ${autoMode ? 'stop' : 'start'}`}
              aria-label={autoMode ? "Stop Navigation" : "Start Navigation"}
            >
              <FontAwesomeIcon 
                icon={autoMode ? faStop : faPlay} 
                className="control-icon" 
              />
              {autoMode ? ' STOP' : ' START'}
            </button>
          </div>

          <div className="voice-selector">
            <label className="voice-label">
              Voice:
            </label>
            <select 
              value={voices.findIndex(v => v === selectedVoice)} 
              onChange={handleVoiceChange} 
              className="voice-dropdown"
              disabled={isSpeaking}
            >
              {voices.map((v, i) => (
                <option key={i} value={i}>
                  {v.name} ({v.lang})
                </option>
              ))}
            </select>
          </div>
          <div className="text-display" aria-live="polite">
            <div className="text-display-content">
              {fetchedText}
            </div>
          </div>

          <div className={`connection-status ${connectionState}`}>
            Status: {connectionState} | Camera: {cameraActive ? 'On' : 'Off'} | Speaking: {isSpeaking ? 'Yes' : 'No'}
          </div>

          <div className="status-indicators">
            <span className={`status-indicator ${cameraActive ? 'status-active' : 'status-inactive'}`}>
              <FontAwesomeIcon 
                icon={cameraActive ? faVideo : faVideoSlash} 
                className="status-icon"
              />
              {cameraActive ? 'Camera On' : 'Camera Off'}
            </span>
            <span className={`status-indicator ${autoMode ? 'status-active' : 'status-inactive'}`}>
              <FontAwesomeIcon 
                icon={autoMode ? faSync : faPause} 
                className="status-icon"
              />
              {autoMode ? 'Auto Mode On' : 'Auto Mode Off'}
            </span>
            <span className={`status-indicator ${isSpeaking ? 'status-speaking' : 'status-inactive'}`}>
              <FontAwesomeIcon 
                icon={isSpeaking ? faMicrophone : faMicrophoneSlash} 
                className="status-icon"
              />
              {isSpeaking ? 'Speaking...' : 'Silent'}
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App