"""Repository addition: typed conversation state scaffold for the starter agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Intent(str, Enum):
    BUYING = "buying"
    BROWSING = "browsing"
    UNCERTAIN = "uncertain"


@dataclass(slots=True)
class UserProfile:
    purchase_frequency: str = ""
    average_prior_rating: float | None = None
    rating_style: str = ""
    preference_tags: tuple[str, ...] = ()
    summary: str = ""

    @classmethod
    def from_payload(cls, payload: dict) -> "UserProfile":
        return cls(
            purchase_frequency=str(payload.get("purchase_frequency", "")),
            average_prior_rating=payload.get("average_prior_rating"),
            rating_style=str(payload.get("rating_style", "")),
            preference_tags=tuple(str(tag) for tag in payload.get("preference_tags", ())),
            summary=str(payload.get("summary", "")),
        )


@dataclass(slots=True)
class SearchConstraints:
    category: str | None = None
    brand: str | None = None
    color: str | None = None
    size: str | None = None
    price_min: float | None = None
    price_max: float | None = None
    gender: str | None = None
    occasion: str | None = None
    style: str | None = None
    material: str | None = None
    hard_constraints: tuple[str, ...] = ()
    soft_preferences: tuple[str, ...] = ()
    negative_preferences: tuple[str, ...] = ()

    def clear_for_new_goal(self) -> None:
        self.category = None
        self.brand = None
        self.color = None
        self.size = None
        self.price_min = None
        self.price_max = None
        self.gender = None
        self.occasion = None
        self.style = None
        self.material = None
        self.hard_constraints = ()
        self.soft_preferences = ()
        self.negative_preferences = ()


@dataclass(slots=True)
class TurnState:
    turn_number: int
    user_message: str
    agent_message: str = ""
    ask_attribute: str | None = None
    recommendations: tuple[str, ...] = ()


@dataclass(slots=True)
class ConversationState:
    session_id: str
    user_profile: UserProfile
    intent: Intent = Intent.UNCERTAIN
    constraints: SearchConstraints = field(default_factory=SearchConstraints)
    turn_number: int = 0
    candidate_count: int | None = None
    current_search_goal: str = ""
    previous_queries: list[str] = field(default_factory=list)
    turns: list[TurnState] = field(default_factory=list)

    @classmethod
    def start(cls, session_id: str, user_profile: dict) -> "ConversationState":
        return cls(session_id=session_id, user_profile=UserProfile.from_payload(user_profile))

    def record_user_turn(self, turn: int, user_message: str) -> TurnState:
        self.turn_number = turn
        self.current_search_goal = user_message
        self.previous_queries.append(user_message)
        turn_state = TurnState(turn_number=turn, user_message=user_message)
        self.turns.append(turn_state)
        return turn_state

    def record_agent_response(
        self,
        message: str,
        ask_attribute: str | None,
        recommendations: list[str],
    ) -> None:
        if not self.turns:
            raise RuntimeError("record_user_turn must be called before record_agent_response")
        current_turn = self.turns[-1]
        current_turn.agent_message = message
        current_turn.ask_attribute = ask_attribute
        current_turn.recommendations = tuple(recommendations)
        self.candidate_count = len(recommendations)

    def begin_new_goal(self, user_message: str, intent: Intent = Intent.UNCERTAIN) -> None:
        """Reset active constraints for an override before the new turn is recorded."""
        self.intent = intent
        self.current_search_goal = user_message
        self.constraints.clear_for_new_goal()
