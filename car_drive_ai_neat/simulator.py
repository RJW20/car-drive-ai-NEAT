from car_drive_ai_neat.player import Player
from car_drive_ai_neat.settings import simulation_settings
from car_drive_app.track.base_track import BaseTrack


def simulate(player: Player) -> Player:
    """Run the player in its environment and assign it a fitness signifying how 
    well it performs.
    
    The track save to use is chosen by simulation_settings['track_name']
    Assigns a fitness that is a ratio of the number of gates cleared and the time 
    taken to clear them.
    """

    track = BaseTrack.load(simulation_settings['track_name'])
    track.place_car_at_start(player)
    
    time = 0
    gates_passed = 0
    current_gate_index = track.current_gate_index
    while track.check_in_bounds(player.outline) and gates_passed < track.total_gates + 2:

        player.look(track)
        player.move(*player.think())
        track.update_gate(player)

        time += 1

        if player.velocity.magnitude < 0.2 and time > 10:
            break

        if track.current_gate_index == (current_gate_index + 1) % track.total_gates:
            gates_passed += 1
        elif track.current_gate_index == (current_gate_index - 1) % track.total_gates:
            gates_passed -= 1
        current_gate_index = track.current_gate_index

    #player.fitness = gates_passed * abs(gates_passed) / time
    player.fitness = gates_passed
    return player