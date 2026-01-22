import os
import sys
# Make current working directory and its parent available for imports (fix relative import error)
# sys.path.insert(0, os.path.abspath(os.getcwd()))
# sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))
from drivingppo.myppo import MyPpoWrapper
from flask import Flask, request, jsonify


modelW = MyPpoWrapper('./1.withobs.zip')

app = Flask(__name__)

info = {}

@app.route('/info', methods=['POST'])
def api_info():
    global info

    data = request.get_json(force=True)
    if not data:
        print('/info 에 데이터가 없다.')
        return jsonify({"error": "No JSON received"}), 400

    info = data

    return jsonify({"status": "success", "control": ""})


def get_player_stat():
    global info


@app.route('/get_action', methods=['POST'])
def api_get_action():
    global info, modelW

    data = request.get_json(force=True)

    position = data.get("position", {})

    pos_x = position.get("x", 0)
    pos_y = position.get("y", 0)
    pos_z = position.get("z", 0)
    info['playerPos']['x'] = pos_x
    info['playerPos']['y'] = pos_y
    info['playerPos']['z'] = pos_z

    # turret = data.get("turret", {})
    # turret_x = turret.get("x", 0)
    # turret_y = turret.get("y", 0)

    act_w, act_s, act_ad = modelW.get_action(info)
    ws_command = 'W' if act_w > act_s else 'S'
    ws_weight  = act_w if act_w > act_s else act_s
    ad_command = 'D'  if act_ad > 0.1  else 'A'  if act_ad < -0.1  else ''
    ad_weight  = act_ad  if act_ad > 0.1  else -act_ad  if act_ad < -0.1  else 0

    return jsonify({
        "moveWS": {"command":ws_command, "weight": ws_weight},
        "moveAD": {"command": ad_command, "weight": ad_weight},
        "turretQE": {"command": "", "weight": 0.0},
        "turretRF": {"command": "", "weight": 0.0},
        "fire": False
    })


@app.route('/set_destination', methods=['POST'])
def api_set_destination():
    data = request.get_json()
    if not data or "destination" not in data:
        return jsonify({"status": "ERROR", "message": "Missing destination data"}), 400

    try:
        x, y, z = map(float, data["destination"].split(","))
        print(f"🎯 Destination set to: x={x}, y={y}, z={z}")
        return jsonify({"status": "OK", "destination": {"x": x, "y": y, "z": z}})
    except Exception as e:
        return jsonify({"status": "ERROR", "message": f"Invalid format: {str(e)}"}), 400


@app.route('/update_obstacle', methods=['POST'])
def api_update_obstacle():
    data = request.get_json()
    if not data:
        return jsonify({'status': 'error', 'message': 'No data received'}), 400

    print("🪨 Obstacle Data:", data)
    return jsonify({'status': 'success', 'message': 'Obstacle data received'})


@app.route('/collision', methods=['POST']) 
def api_collision():
    data = request.get_json()
    if not data:
        return jsonify({'status': 'error', 'message': 'No collision data received'}), 400

    object_name = data.get('objectName')
    position = data.get('position', {})
    x = position.get('x')
    y = position.get('y')
    z = position.get('z')

    print(f"💥 Collision Detected - Object: {object_name}, Position: ({x}, {y}, {z})")

    return jsonify({'status': 'success', 'message': 'Collision data received'})


@app.route('/init', methods=['GET'])
def api_init():
    global info, modelW

    x = 1
    y = 10
    z = 2

    config = {
        "startMode": "start",  # Options: "start" or "pause"

        "blStartX": x,
        "blStartY": y,
        "blStartZ": z,

        "rdStartX": 59,
        "rdStartY": 10,
        "rdStartZ": 280,

        "trackingMode": True,
        "detectMode": False,
        "logMode": True,
        "stereoCameraMode": False,
        "enemyTracking": False,

        "saveSnapshot": False,
        "saveLog": False,
        "saveLidarData": False,
        "lux": 30000,
        "destoryObstaclesOnHit" : True
    }

    modelW.init(config)

    print("🛠️ INIT", config)
    return jsonify(config)


@app.route('/start', methods=['GET'])
def api_start():
    # print("🚀 /start command received")
    return jsonify({"control": ""})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
