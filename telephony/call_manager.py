"""
Call management for Twilio integration.
"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import asyncio

from telephony.config import MAX_CALL_DURATION

logger = logging.getLogger(__name__)

class CallManager:
    """
    Manages active calls and their states.
    """
    
    def __init__(self):
        """Initialize call manager."""
        self.active_calls: Dict[str, Dict[str, Any]] = {}
        self._cleanup_task = None
    
    async def start(self):
        """Start the call manager."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Call manager started")
    
    async def stop(self):
        """Stop the call manager."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("Call manager stopped")
    
    def add_call(self, call_sid: str, from_number: str, to_number: str) -> None:
        """
        Add a new call to tracking.
        
        Args:
            call_sid: Twilio call SID
            from_number: Caller phone number
            to_number: Called phone number
        """
        self.active_calls[call_sid] = {
            'call_sid': call_sid,
            'from_number': from_number,
            'to_number': to_number,
            'start_time': datetime.now(),
            'status': 'active',
            'stream_sid': None,
            'transcription': '',
            'response': '',
            'conversation_history': []
        }
        logger.info(f"Added call {call_sid} from {from_number}")
    
    def update_call_status(self, call_sid: str, status: str) -> None:
        """
        Update call status.
        
        Args:
            call_sid: Twilio call SID
            status: New status
        """
        if call_sid in self.active_calls:
            self.active_calls[call_sid]['status'] = status
            logger.info(f"Updated call {call_sid} status to {status}")
    
    def set_stream_sid(self, call_sid: str, stream_sid: str) -> None:
        """
        Set stream SID for a call.
        
        Args:
            call_sid: Twilio call SID
            stream_sid: Twilio stream SID
        """
        if call_sid in self.active_calls:
            self.active_calls[call_sid]['stream_sid'] = stream_sid
            logger.info(f"Set stream {stream_sid} for call {call_sid}")
    
    def add_conversation_turn(self, call_sid: str, speaker: str, text: str) -> None:
        """
        Add a conversation turn to call history.
        
        Args:
            call_sid: Twilio call SID
            speaker: 'user' or 'assistant'
            text: Transcription or response text
        """
        if call_sid in self.active_calls:
            turn = {
                'speaker': speaker,
                'text': text,
                'timestamp': datetime.now().isoformat()
            }
            self.active_calls[call_sid]['conversation_history'].append(turn)
            
            # Update transcription or response
            if speaker == 'user':
                self.active_calls[call_sid]['transcription'] = text
            else:
                self.active_calls[call_sid]['response'] = text
    
    def get_call(self, call_sid: str) -> Optional[Dict[str, Any]]:
        """
        Get call information.
        
        Args:
            call_sid: Twilio call SID
            
        Returns:
            Call information or None if not found
        """
        return self.active_calls.get(call_sid)
    
    def remove_call(self, call_sid: str) -> None:
        """
        Remove a call from tracking.
        
        Args:
            call_sid: Twilio call SID
        """
        if call_sid in self.active_calls:
            call_info = self.active_calls[call_sid]
            duration = (datetime.now() - call_info['start_time']).total_seconds()
            logger.info(f"Removing call {call_sid} after {duration:.1f}s")
            
            # Log conversation history
            self._log_conversation(call_info)
            
            del self.active_calls[call_sid]
    
    def _log_conversation(self, call_info: Dict[str, Any]) -> None:
        """Log conversation history for a call."""
        logger.info(f"Conversation history for call {call_info['call_sid']}:")
        for turn in call_info['conversation_history']:
            logger.info(f"  {turn['speaker']}: {turn['text']}")
    
    async def _cleanup_loop(self) -> None:
        """Periodically clean up old calls."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                current_time = datetime.now()
                
                # Find calls to remove
                calls_to_remove = []
                for call_sid, call_info in self.active_calls.items():
                    duration = (current_time - call_info['start_time']).total_seconds()
                    
                    # Remove if call is too old or completed
                    if duration > MAX_CALL_DURATION or call_info['status'] in ['completed', 'failed']:
                        calls_to_remove.append(call_sid)
                
                # Remove old calls
                for call_sid in calls_to_remove:
                    self.remove_call(call_sid)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    def get_active_call_count(self) -> int:
        """Get number of active calls."""
        return len(self.active_calls)
    
    def get_call_stats(self) -> Dict[str, Any]:
        """Get statistics about calls."""
        total_calls = len(self.active_calls)
        active_calls = sum(1 for call in self.active_calls.values() if call['status'] == 'active')
        
        return {
            'total_calls': total_calls,
            'active_calls': active_calls,
            'calls_by_status': self._count_by_status()
        }
    
    def _count_by_status(self) -> Dict[str, int]:
        """Count calls by status."""
        status_count = {}
        for call in self.active_calls.values():
            status = call['status']
            status_count[status] = status_count.get(status, 0) + 1
        return status_count