from typing import Callable, List, Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class FLX4State:
    crossfader: float = 0.5         # 0.0 (Deck 1) to 1.0 (Deck 2)
    deck1_volume: float = 1.0       # 0.0 to 1.0
    deck2_volume: float = 1.0       # 0.0 to 1.0
    deck1_eq_high: float = 0.5      # 0.0 to 1.0 (0.5 = 12 o'clock center)
    deck1_eq_mid: float = 0.5
    deck1_eq_low: float = 0.5
    deck2_eq_high: float = 0.5
    deck2_eq_mid: float = 0.5
    deck2_eq_low: float = 0.5
    deck1_playing: bool = False
    deck2_playing: bool = False
    deck1_hotcue: Optional[int] = None
    deck2_hotcue: Optional[int] = None
    master_deck: int = 1            # Computed: 1 or 2


class FLX4Controller:
    """
    Hardware MIDI protocol parser and state engine for Pioneer DDJ-FLX4 USB DJ controller.
    """

    # Pioneer DDJ-FLX4 standard MIDI mapping specs
    CC_CROSSFADER = 31
    CC_DECK1_VOL = 19
    CC_DECK2_VOL = 20
    CC_DECK1_EQ_HIGH = 7
    CC_DECK1_EQ_MID = 11
    CC_DECK1_EQ_LOW = 15
    CC_DECK2_EQ_HIGH = 8
    CC_DECK2_EQ_MID = 12
    CC_DECK2_EQ_LOW = 16

    NOTE_DECK1_PLAY = 11
    NOTE_DECK2_PLAY = 12
    NOTE_DECK1_CUE = 13
    NOTE_DECK2_CUE = 14

    PADS_DECK1 = list(range(0x00, 0x08))  # Pads 1-8
    PADS_DECK2 = list(range(0x08, 0x10))  # Pads 1-8

    def __init__(self):
        self.state = FLX4State()
        self._listeners: List[Callable[[FLX4State, str], None]] = []

    def register_listener(self, callback: Callable[[FLX4State, str], None]) -> None:
        """Register a callback that fires whenever hardware state changes."""
        self._listeners.append(callback)

    def _notify(self, event_name: str) -> None:
        self._update_master_deck()
        for listener in self._listeners:
            listener(self.state, event_name)

    def _update_master_deck(self) -> None:
        # Determine dominant audible deck
        d1_audible = self.state.deck1_volume * (1.0 - self.state.crossfader)
        d2_audible = self.state.deck2_volume * self.state.crossfader
        self.state.master_deck = 1 if d1_audible >= d2_audible else 2

    def parse_midi_message(self, status: int, data1: int, data2: int) -> Optional[str]:
        """
        Parses a 3-byte MIDI event from the DDJ-FLX4 USB endpoint.
        Returns event name if recognized.
        """
        msg_type = status & 0xF0

        # Control Change (Faders, Knobs)
        if msg_type == 0xB0:
            val_norm = data2 / 127.0

            if data1 == self.CC_CROSSFADER:
                self.state.crossfader = round(val_norm, 3)
                self._notify("crossfader_move")
                return "crossfader_move"
            elif data1 == self.CC_DECK1_VOL:
                self.state.deck1_volume = round(val_norm, 3)
                self._notify("deck1_vol_change")
                return "deck1_vol_change"
            elif data1 == self.CC_DECK2_VOL:
                self.state.deck2_volume = round(val_norm, 3)
                self._notify("deck2_vol_change")
                return "deck2_vol_change"
            elif data1 == self.CC_DECK1_EQ_LOW:
                self.state.deck1_eq_low = round(val_norm, 3)
                self._notify("deck1_eq_low_change")
                return "deck1_eq_low_change"
            elif data1 == self.CC_DECK2_EQ_LOW:
                self.state.deck2_eq_low = round(val_norm, 3)
                self._notify("deck2_eq_low_change")
                return "deck2_eq_low_change"

        # Note On (Buttons, Pads)
        elif msg_type == 0x90 and data2 > 0:
            if data1 == self.NOTE_DECK1_PLAY:
                self.state.deck1_playing = not self.state.deck1_playing
                self._notify("deck1_play_toggle")
                return "deck1_play_toggle"
            elif data1 == self.NOTE_DECK2_PLAY:
                self.state.deck2_playing = not self.state.deck2_playing
                self._notify("deck2_play_toggle")
                return "deck2_play_toggle"
            elif data1 in self.PADS_DECK1:
                pad_num = self.PADS_DECK1.index(data1) + 1
                self.state.deck1_hotcue = pad_num
                self._notify(f"deck1_pad_{pad_num}")
                return f"deck1_pad_{pad_num}"
            elif data1 in self.PADS_DECK2:
                pad_num = self.PADS_DECK2.index(data1) + 1
                self.state.deck2_hotcue = pad_num
                self._notify(f"deck2_pad_{pad_num}")
                return f"deck2_pad_{pad_num}"

        return None

    def update_faders(self, crossfader: float, deck1_vol: float, deck2_vol: float) -> None:
        """Utility for software/web GUI simulation of fader changes."""
        self.state.crossfader = max(0.0, min(1.0, crossfader))
        self.state.deck1_volume = max(0.0, min(1.0, deck1_vol))
        self.state.deck2_volume = max(0.0, min(1.0, deck2_vol))
        self._notify("fader_sync")
