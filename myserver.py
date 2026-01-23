"""
Tank Challenge API
https://bangbaedong-vallet-co-ltd.gitbook.io/tank-challenge/3.-api/3.2-api-docs
"""
from drivingppo.adaptor import MyPpoAdaptor
from flask import Flask, request, jsonify


modelW = MyPpoAdaptor('./ppo_world_checkpoints/drivingppo.zip')

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

    stop, ws, ad = modelW.get_action(info)

    ws_command = "W"  if ws > 0.0  else "S"
    ws_weight  = ws   if ws > 0.0  else -ws
    ad_command = "D"  if ad > 0.0  else "A"
    ad_weight  = ad   if ad > 0.0  else -ad
    if stop:
        ws_command = "STOP"
        ws_weight  = 0.0

    return jsonify({
        "moveWS": {"command": ws_command, "weight": ws_weight},
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
        print(f"🚩 목적지 설정 ({x:.1f}, {y:.1f}, {z:.1f})")
        return jsonify({"status": "OK", "destination": {"x": x, "y": y, "z": z}})
    except Exception as e:
        return jsonify({"status": "ERROR", "message": f"Invalid format: {str(e)}"}), 400


@app.route('/update_obstacle', methods=['POST'])
def api_update_obstacle():
    data = request.get_json()
    if not data:
        return jsonify({'status': 'error', 'message': 'No data received'}), 400

    obss:list = data['obstacles']

    if obss:
        obs = obss[-1]
        print(f"🪨 장애물 {len(obss)} 개  | 마지막거 위치: ({int(round(obs['x_min']))}~{int(round(obs['x_max']))}, {int(round(obs['z_min']))}~{int(round(obs['z_max']))})")
    else:
        print('🪨 장애물 없음')
    return jsonify({"status": "success", "message": "Obstacle data received"})


@app.route('/collision', methods=['POST']) 
def api_collision():
    data = request.get_json()
    if not data:
        return jsonify({'status': 'error', 'message': 'No collision data received'}), 400

    object_name = data.get("objectName")
    position = data.get("position", {})
    x = position.get("x")
    y = position.get("y")
    z = position.get("z")

    print(f"💥 {object_name}와 충돌: (위치: {x:.1f}, {y:.1f}, {z:.1f})")

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
