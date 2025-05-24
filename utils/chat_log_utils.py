import json
from datetime import datetime

def load_sessions(file_path="chat_log.json"):
    with open(file_path, "r") as f:
        data = json.load(f)

    sessions = []
    current_session = []
    for entry in data:
        if entry.get("event") == "session_start":
            current_session = [{"type":"event","event":"start", "time": entry["time"]}]
        elif entry.get("event") == "session_end":
            current_session.append({"type":"event", "event":"end", "time": entry["time"]})
            sessions.append(current_session)
        else:
            current_session.append({
                "type":"message",
                "user": entry["user"],
                "assistant": entry["assistant"],
                "time": entry["time"]
            })
    return sessions

def get_session_summaries(sessions):
    return [f"Chat on {datetime.strptime(session[0]['time'], '%Y-%m-%d %H:%M:%S').strftime('%b %d, %Y %I:%M %p')}" for session in sessions]