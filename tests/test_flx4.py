from sonicdj.hardware.flx4 import FLX4Controller


def test_flx4_fader_and_eq_parsing():
    controller = FLX4Controller()
    events = []

    def on_event(state, event_name):
        events.append(event_name)

    controller.register_listener(on_event)

    # 1. Move Crossfader to 0.0 (Deck 1 hard left)
    # 0xB0 (CC) 31 (Crossfader) 0 (Left)
    ev1 = controller.parse_midi_message(0xB0, 31, 0)
    assert ev1 == "crossfader_move"
    assert controller.state.crossfader == 0.0
    assert controller.state.master_deck == 1

    # 2. Move Crossfader to 127 (Deck 2 hard right)
    ev2 = controller.parse_midi_message(0xB0, 31, 127)
    assert ev2 == "crossfader_move"
    assert controller.state.crossfader == 1.0
    assert controller.state.master_deck == 2

    # 3. Channel Volumes
    controller.parse_midi_message(0xB0, 19, 100) # Deck 1 Vol
    assert controller.state.deck1_volume > 0.7

    controller.parse_midi_message(0xB0, 20, 50)  # Deck 2 Vol
    assert controller.state.deck2_volume < 0.5

    # 4. Low EQ Knobs
    controller.parse_midi_message(0xB0, 15, 64)  # D1 Low EQ
    assert abs(controller.state.deck1_eq_low - 0.5) < 0.02


def test_flx4_buttons_and_pads():
    controller = FLX4Controller()

    # 1. Play Button Deck 1 (Note 11)
    ev_play = controller.parse_midi_message(0x90, 11, 127)
    assert ev_play == "deck1_play_toggle"
    assert controller.state.deck1_playing is True

    # 2. Performance Pad 3 on Deck 1 (Note 0x02)
    ev_pad = controller.parse_midi_message(0x90, 0x02, 127)
    assert ev_pad == "deck1_pad_3"
    assert controller.state.deck1_hotcue == 3

    # 3. Performance Pad 1 on Deck 2 (Note 0x08)
    ev_pad2 = controller.parse_midi_message(0x90, 0x08, 127)
    assert ev_pad2 == "deck2_pad_1"
    assert controller.state.deck2_hotcue == 1
