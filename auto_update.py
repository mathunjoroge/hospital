#!/usr/bin/env python3
"""
Complete Telemedicine Feature Implementation Script
Run this in your local hospital repository to implement all missing features
"""

import os
import sys

def main():
    print("=" * 70)
    print("TELEMEDICINE FEATURE IMPLEMENTATION")
    print("=" * 70)
    
    # Check if we're in the right directory
    if not os.path.exists('departments/telemedicine'):
        print("❌ Error: Please run this script from the hospital repository root")
        print("   Expected to find: departments/telemedicine/")
        sys.exit(1)
    
    # 1. Implement WebRTC in room.html
    print("\n1. Implementing WebRTC in room.html...")
    room_html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Telemedicine Consultation — {{ session.session_uuid[:8] }}</title>
  <meta name="csrf-token" content="{{ csrf_token() }}">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
  <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
  <style>
    body { background-color: #0f172a; color: #f8fafc; font-family: system-ui, sans-serif; }
    .video-canvas { background-color: #1e293b; border-radius: 12px; height: 380px; position: relative; overflow: hidden; }
    .remote-video { width: 100%; height: 100%; object-fit: cover; background: #000; }
    .local-video { width: 120px; height: 90px; position: absolute; bottom: 12px; right: 12px; border-radius: 8px; border: 2px solid #38bdf8; background: #000; }
    .card-panel { background-color: #1e293b; border: 1px solid #334155; border-radius: 12px; }
    .connection-status { position: absolute; top: 12px; left: 12px; z-index: 10; }
    .status-badge { padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
    .status-connecting { background: #fbbf24; color: #000; }
    .status-connected { background: #10b981; color: #fff; }
    .status-error { background: #ef4444; color: #fff; }
  </style>
</head>
<body class="p-3">
  <div class="container-fluid">
    <div class="d-flex justify-content-between align-items-center mb-3">
      <div>
        <h4 class="mb-0 text-info">🩺 Virtual Telemedicine Consultation</h4>
        <small class="text-secondary">Session ID: {{ session.session_uuid }} | Patient: {{ session.patient_id }}</small>
      </div>
      <div>
        <span class="badge bg-success fs-6" id="session-status">{{ session.status }}</span>
      </div>
    </div>

    <div class="row g-3">
      <div class="col-lg-7">
        <div class="video-canvas d-flex justify-content-center align-items-center">
          <div class="connection-status">
            <span id="connection-badge" class="status-badge status-connecting">Connecting...</span>
          </div>
          <div class="text-center" id="video-placeholder">
            <div class="fs-1 mb-2">📹</div>
            <h5>Initializing Video Connection...</h5>
            <p class="text-secondary small">Room Token: {{ session.room_token }}</p>
          </div>
          <video id="remote-stream" class="remote-video d-none" autoplay playsinline></video>
          <video id="local-stream" class="local-video d-none" autoplay playsinline muted></video>
        </div>

        <div class="d-flex justify-content-center gap-2 mt-3">
          <button class="btn btn-outline-light" id="btn-toggle-cam" onclick="toggleCamera()">📷 Toggle Camera</button>
          <button class="btn btn-outline-light" id="btn-toggle-mic" onclick="toggleMicrophone()">🎙️ Toggle Mic</button>
          <button class="btn btn-danger" id="btn-end-call" onclick="completeSession()">🛑 End Consultation</button>
        </div>
      </div>

      <div class="col-lg-5">
        <div class="card-panel p-3">
          <h5 class="text-light mb-3">📝 Doctor Consultation Notes</h5>
          <textarea id="notes-input" class="form-control bg-dark text-light border-secondary mb-3" rows="10" placeholder="Type clinical findings...">{{ session.clinical_notes or '' }}</textarea>
          <div class="d-flex justify-content-between">
            <button class="btn btn-primary" onclick="saveNotes()">💾 Save Notes</button>
            <button class="btn btn-success" onclick="completeSession()">✅ Finalize Consultation</button>
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    const sessionUuid = "{{ session.session_uuid }}";
    const roomToken = "{{ session.room_token }}";
    let localStream = null;
    let peerConnection = null;
    let socket = null;

    const rtcConfig = {
      iceServers: [
        { urls: 'stun:stun.l.google.com:19302' },
        { urls: 'stun:stun1.l.google.com:19302' }
      ]
    };

    document.addEventListener('DOMContentLoaded', async () => {
      await initializeCall();
    });

    async function initializeCall() {
      try {
        updateConnectionStatus('connecting', 'Initializing...');
        localStream = await navigator.mediaDevices.getUserMedia({
          video: { width: 1280, height: 720 },
          audio: true
        });

        const localVideo = document.getElementById('local-stream');
        localVideo.srcObject = localStream;
        localVideo.classList.remove('d-none');
        document.getElementById('video-placeholder').classList.add('d-none');

        socket = io();
        socket.emit('join_room', { room_token: roomToken, session_uuid: sessionUuid });

        peerConnection = new RTCPeerConnection(rtcConfig);
        localStream.getTracks().forEach(track => {
          peerConnection.addTrack(track, localStream);
        });

        peerConnection.ontrack = (event) => {
          const remoteVideo = document.getElementById('remote-stream');
          remoteVideo.srcObject = event.streams[0];
          remoteVideo.classList.remove('d-none');
          updateConnectionStatus('connected', 'Connected');
        };

        peerConnection.onicecandidate = (event) => {
          if (event.candidate) {
            socket.emit('ice_candidate', { room_token: roomToken, candidate: event.candidate });
          }
        };

        socket.on('user_joined', async (data) => {
          if (data.is_initiator) {
            await createOffer();
          }
        });

        socket.on('offer', async (data) => {
          await peerConnection.setRemoteDescription(new RTCSessionDescription(data.offer));
          const answer = await peerConnection.createAnswer();
          await peerConnection.setLocalDescription(answer);
          socket.emit('answer', { room_token: roomToken, answer: answer });
        });

        socket.on('answer', async (data) => {
          await peerConnection.setRemoteDescription(new RTCSessionDescription(data.answer));
        });

        socket.on('ice_candidate', async (data) => {
          try {
            await peerConnection.addIceCandidate(new RTCIceCandidate(data.candidate));
          } catch (e) {
            console.error('Error adding ICE candidate:', e);
          }
        });

        updateConnectionStatus('connecting', 'Waiting for peer...');
      } catch (error) {
        console.error('Error initializing call:', error);
        updateConnectionStatus('error', 'Error: ' + error.message);
      }
    }

    async function createOffer() {
      const offer = await peerConnection.createOffer();
      await peerConnection.setLocalDescription(offer);
      socket.emit('offer', { room_token: roomToken, offer: offer });
    }

    function toggleCamera() {
      if (localStream) {
        const videoTrack = localStream.getVideoTracks()[0];
        videoTrack.enabled = !videoTrack.enabled;
      }
    }

    function toggleMicrophone() {
      if (localStream) {
        const audioTrack = localStream.getAudioTracks()[0];
        audioTrack.enabled = !audioTrack.enabled;
      }
    }

    function updateConnectionStatus(status, message) {
      const badge = document.getElementById('connection-badge');
      badge.className = 'status-badge status-' + status;
      badge.textContent = message;
    }

    async function saveNotes() {
      const notes = document.getElementById('notes-input').value;
      const csrfToken = document.querySelector('meta[name="csrf-token"]')?.content || '';
      await fetch(`/telemedicine/session/${sessionUuid}/notes`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-CSRFToken': csrfToken },
        body: JSON.stringify({ notes: notes })
      });
      alert("Notes saved!");
    }

    async function completeSession() {
      if (!confirm('End consultation?')) return;
      const notes = document.getElementById('notes-input').value;
      const csrfToken = document.querySelector('meta[name="csrf-token"]')?.content || '';
      const res = await fetch(`/telemedicine/session/${sessionUuid}/complete`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-CSRFToken': csrfToken },
        body: JSON.stringify({ notes: notes })
      });
      if (res.ok) {
        if (peerConnection) peerConnection.close();
        if (localStream) localStream.getTracks().forEach(track => track.stop());
        if (socket) socket.disconnect();
        alert("Consultation finalized.");
        window.location.href = "/";
      }
    }
  </script>
</body>
</html>"""

    with open('departments/telemedicine/templates/telemedicine/room.html', 'w') as f:
        f.write(room_html)
    print("   ✅ WebRTC implementation complete")

    # 2. Add Socket.IO signaling
    print("\n2. Adding Socket.IO signaling...")
    signaling = """from flask_socketio import emit, join_room, leave_room
from flask import request
from extensions import socketio

active_rooms = {}

@socketio.on('join_room')
def handle_join_room(data):
    room_token = data.get('room_token')
    if room_token:
        join_room(room_token)
        if room_token not in active_rooms:
            active_rooms[room_token] = []
        is_initiator = len(active_rooms[room_token]) == 0
        active_rooms[room_token].append(request.sid)
        emit('user_joined', {'sid': request.sid, 'is_initiator': is_initiator}, room=room_token)

@socketio.on('leave_room')
def handle_leave_room(data):
    room_token = data.get('room_token')
    if room_token:
        leave_room(room_token)
        if room_token in active_rooms and request.sid in active_rooms[room_token]:
            active_rooms[room_token].remove(request.sid)

@socketio.on('offer')
def handle_offer(data):
    room_token = data.get('room_token')
    if room_token:
        emit('offer', {'offer': data.get('offer')}, room=room_token, include_self=False)

@socketio.on('answer')
def handle_answer(data):
    room_token = data.get('room_token')
    if room_token:
        emit('answer', {'answer': data.get('answer')}, room=room_token, include_self=False)

@socketio.on('ice_candidate')
def handle_ice_candidate(data):
    room_token = data.get('room_token')
    if room_token:
        emit('ice_candidate', {'candidate': data.get('candidate')}, room=room_token, include_self=False)

@socketio.on('disconnect')
def handle_disconnect():
    for room_token, participants in list(active_rooms.items()):
        if request.sid in participants:
            participants.remove(request.sid)
            if not participants:
                del active_rooms[room_token]
"""

    with open('departments/telemedicine/signaling.py', 'w') as f:
        f.write(signaling)
    
    # Register signaling in __init__.py
    with open('departments/telemedicine/__init__.py', 'r') as f:
        init_content = f.read()
    if 'signaling' not in init_content:
        init_content += "\nfrom . import signaling\n"
        with open('departments/telemedicine/__init__.py', 'w') as f:
            f.write(init_content)
    
    print("   ✅ Socket.IO signaling server created")

    # 3. Create sessions list page
    print("\n3. Creating sessions list page...")
    os.makedirs('departments/telemedicine/templates/telemedicine', exist_ok=True)
    sessions_html = """{% extends "base.html" %}
{% block content %}
<div class="container-fluid p-4">
  <h2 class="mb-4">📹 Telemedicine Sessions</h2>
  
  <div class="row mb-4">
    <div class="col-md-6">
      <div class="card">
        <div class="card-body">
          <h5 class="card-title">Start New Consultation</h5>
          <button class="btn btn-primary" data-bs-toggle="modal" data-bs-target="#createSessionModal">
            <i class="fas fa-plus"></i> Create Session
          </button>
        </div>
      </div>
    </div>
  </div>

  <div class="card">
    <div class="card-header"><h5 class="mb-0">Your Sessions</h5></div>
    <div class="card-body">
      <table class="table table-hover">
        <thead>
          <tr>
            <th>Session ID</th>
            <th>Patient</th>
            <th>Status</th>
            <th>Created</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody id="sessions-tbody">
          <tr><td colspan="5" class="text-center">Loading...</td></tr>
        </tbody>
      </table>
    </div>
  </div>
</div>

<div class="modal fade" id="createSessionModal" tabindex="-1">
  <div class="modal-dialog">
    <div class="modal-content">
      <div class="modal-header">
        <h5 class="modal-title">Create Telemedicine Session</h5>
        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
      </div>
      <div class="modal-body">
        <div class="mb-3">
          <label class="form-label">Patient ID</label>
          <input type="text" class="form-control" id="patient_id" required>
        </div>
      </div>
      <div class="modal-footer">
        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
        <button type="button" class="btn btn-primary" onclick="createSession()">Create</button>
      </div>
    </div>
  </div>
</div>

<script>
document.addEventListener('DOMContentLoaded', loadSessions);

async function loadSessions() {
  const res = await fetch('/telemedicine/sessions');
  const data = await res.json();
  const tbody = document.getElementById('sessions-tbody');
  tbody.innerHTML = '';
  
  if (data.sessions && data.sessions.length > 0) {
    data.sessions.forEach(session => {
      const row = document.createElement('tr');
      row.innerHTML = `
        <td>${session.session_uuid.substring(0, 8)}...</td>
        <td>${session.patient_id}</td>
        <td><span class="badge bg-${session.status === 'ACTIVE' ? 'success' : 'secondary'}">${session.status}</span></td>
        <td>${new Date(session.created_at).toLocaleString()}</td>
        <td>
          ${session.status !== 'COMPLETED' ? 
            `<a href="/telemedicine/room/${session.session_uuid}" class="btn btn-sm btn-success">Join</a>` : 
            '<span class="text-muted">Completed</span>'}
        </td>
      `;
      tbody.appendChild(row);
    });
  } else {
    tbody.innerHTML = '<tr><td colspan="5" class="text-center">No sessions found</td></tr>';
  }
}

async function createSession() {
  const patient_id = document.getElementById('patient_id').value;
  if (!patient_id) return alert('Patient ID required');
  
  const res = await fetch('/telemedicine/session/create', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ patient_id: patient_id })
  });
  
  if (res.ok) {
    alert('Session created!');
    bootstrap.Modal.getInstance(document.getElementById('createSessionModal')).hide();
    loadSessions();
  } else {
    alert('Error creating session');
  }
}
</script>
{% endblock %}"""

    with open('departments/telemedicine/templates/telemedicine/sessions.html', 'w') as f:
        f.write(sessions_html)
    
    # Add route for sessions page
    with open('departments/telemedicine/routes.py', 'r') as f:
        routes = f.read()
    
    if '/sessions/list' not in routes:
        routes += '''

@bp.route("/sessions/list", methods=["GET"])
@login_required
def sessions_list():
    """Render the sessions list page."""
    return render_template("telemedicine/sessions.html")
'''
        with open('departments/telemedicine/routes.py', 'w') as f:
            f.write(routes)
    
    print("   ✅ Sessions list page created")

    # 4. Add navigation links
    print("\n4. Adding navigation links...")
    nav_templates = [
        'templates/side_bars/doctor.html',
        'templates/side_bars/medicine.html'
    ]
    
    nav_link = '''
    <li class="nav-item">
      <a class="nav-link" href="/telemedicine/sessions/list">
        <i class="fas fa-video"></i>
        <span>Telemedicine</span>
      </a>
    </li>
'''
    
    for template in nav_templates:
        if os.path.exists(template):
            with open(template, 'r') as f:
                content = f.read()
            if 'telemedicine' not in content.lower():
                if '</ul>' in content:
                    content = content.replace('</ul>', nav_link + '\n</ul>', 1)
                    with open(template, 'w') as f:
                        f.write(content)
                    print(f"   ✅ Navigation added to {template}")
                    break

    # 5. Enable feature flag
    print("\n5. Enabling feature flag...")
    if os.path.exists('.env.example'):
        with open('.env.example', 'r') as f:
            env = f.read()
        if 'ENABLE_TELEMEDICINE' not in env:
            env += "\n# Enable telemedicine feature\nENABLE_TELEMEDICINE=true\n"
            with open('.env.example', 'w') as f:
                f.write(env)
            print("   ✅ Feature flag added to .env.example")

    print("\n" + "=" * 70)
    print("IMPLEMENTATION COMPLETE!")
    print("=" * 70)
    print("\n✅ All missing telemedicine features implemented:")
    print("   1. WebRTC video calling with peer-to-peer connection")
    print("   2. Socket.IO signaling server for ICE/SDP exchange")
    print("   3. Camera and microphone toggle controls")
    print("   4. Navigation links in doctor dashboard")
    print("   5. Sessions list page with create/join functionality")
    print("   6. Feature flag enabled")
    print("\n📋 Next steps:")
    print("   1. pip install flask-socketio")
    print("   2. Add ENABLE_TELEMEDICINE=true to your .env file")
    print("   3. Restart Flask application")
    print("   4. Navigate to /telemedicine/sessions/list")
    print("\n📋 To commit these changes:")
    print("   git add -A")
    print("   git commit -m 'feat: implement complete telemedicine with WebRTC'")
    print("   git push origin main")

if __name__ == "__main__":
    main()