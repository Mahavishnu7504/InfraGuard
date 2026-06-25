from backend.services.event_service import save_event

def save_safety_log(data):
    return save_event(data)