from flask_socketio import emit, join_room, leave_room
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
