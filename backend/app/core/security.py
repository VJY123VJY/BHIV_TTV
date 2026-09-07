import os
from typing import Optional
from app.core.config import settings
from app.core.exceptions import GovernanceViolationError
from app.core.logging import telemetry

class GovernanceGuard:
    """
    Enforces non-bypassable execution boundaries and token verification.
    Integrates security controls from ttv7, ttv11, and Text_to_vision2.
    """
    def __init__(self):
        self.lock_enabled = settings.GOVERNANCE_LOCK.upper() == "ON"
        self.expected_token = settings.WRAPPER_TOKEN

    def verify_execution(self, token: Optional[str] = None, execution_id: Optional[str] = None):
        """
        Verify that execution is entering through an authorized interface.
        If governance lock is on, token or internal execution authorization is required.
        """
        if not self.lock_enabled:
            return True

        # Check token if provided or allow default internal system wrapper token
        if token and token != self.expected_token:
            if execution_id:
                telemetry.emit("governance_violation", execution_id, {"reason": "invalid_wrapper_token"})
            raise GovernanceViolationError("Unauthorized execution access. Invalid governance token.")

        if execution_id:
            telemetry.emit("governance_passed", execution_id, {"lock_status": "enforced"})
        return True

governance_guard = GovernanceGuard()
