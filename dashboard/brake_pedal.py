import streamlit as st


BRAKE_HTML = """
<div class="brake-wrapper">
    <div id="brake-pedal" class="brake-pedal">
        <div class="pedal-icon">🦶</div>
        <div class="pedal-title">BRAKE</div>
        <div id="pedal-status" class="pedal-status">
            RELEASED
        </div>
        <div id="pedal-pressure" class="pedal-pressure">
            0%
        </div>
    </div>
</div>
"""


BRAKE_CSS = """
.brake-wrapper {
    width: 100%;
    height: 330px;

    display: flex;
    justify-content: center;
    align-items: center;
}

.brake-pedal {
    width: 190px;
    height: 270px;

    border-radius: 30px;

    background: linear-gradient(
        180deg,
        #444444,
        #151515
    );

    border: 5px solid #777777;

    box-shadow:
        0 18px 30px rgba(0,0,0,0.50),
        inset 0 4px 8px rgba(255,255,255,0.12);

    display: flex;
    flex-direction: column;

    justify-content: center;
    align-items: center;

    color: white;

    cursor: pointer;

    user-select: none;

    touch-action: none;

    transition:
        transform 0.08s ease,
        border-color 0.08s ease,
        box-shadow 0.08s ease;
}

.brake-pedal:hover {
    border-color: #999999;
}

.brake-pedal.pressed {
    transform: translateY(10px) scale(0.97);

    border-color: #ff4444;

    box-shadow:
        0 5px 12px rgba(0,0,0,0.50),
        inset 0 5px 10px rgba(0,0,0,0.50);
}

.pedal-icon {
    font-size: 58px;
}

.pedal-title {
    margin-top: 10px;

    font-size: 27px;

    font-weight: 700;
}

.pedal-status {
    margin-top: 12px;

    font-size: 17px;

    font-weight: 600;
}

.pedal-pressure {
    margin-top: 5px;

    font-size: 15px;

    opacity: 0.8;
}
"""


BRAKE_JS = """
export default function(component) {

    const {
        parentElement,
        setStateValue
    } = component;

    const pedal =
        parentElement.querySelector("#brake-pedal");

    const status =
        parentElement.querySelector("#pedal-status");

    const pressure =
        parentElement.querySelector("#pedal-pressure");


    let braking = false;


    function clamp(value) {

        return Math.max(
            0,
            Math.min(100, value)
        );

    }


    function calculatePressure(clientY) {

        const rect =
            pedal.getBoundingClientRect();

        /*
        Top of pedal = 0%
        Bottom of pedal = 100%
        */

        let value =
            ((clientY - rect.top) / rect.height) * 100;

        return clamp(value);

    }


    function updatePedal(value) {

        value = clamp(value);

        setStateValue(
            "pressure",
            value
        );

        pressure.innerText =
            Math.round(value) + "%";


        if (value > 0) {

            status.innerText =
                "BRAKING";

            pedal.classList.add(
                "pressed"
            );

        } else {

            status.innerText =
                "RELEASED";

            pedal.classList.remove(
                "pressed"
            );

        }

    }


    function startBrake(event) {

        event.preventDefault();

        braking = true;

        pedal.setPointerCapture(
            event.pointerId
        );

        updatePedal(
            calculatePressure(
                event.clientY
            )
        );

    }


    function moveBrake(event) {

        if (!braking) {
            return;
        }

        updatePedal(
            calculatePressure(
                event.clientY
            )
        );

    }


    function stopBrake() {

        if (!braking) {
            return;
        }

        braking = false;

        updatePedal(0);

    }


    pedal.addEventListener(
        "pointerdown",
        startBrake
    );


    pedal.addEventListener(
        "pointermove",
        moveBrake
    );


    pedal.addEventListener(
        "pointerup",
        stopBrake
    );


    pedal.addEventListener(
        "pointercancel",
        stopBrake
    );


    /*
    Keyboard support
    */

    document.addEventListener(
        "keydown",
        function(event) {

            if (
                event.code === "Space" &&
                !event.repeat
            ) {

                event.preventDefault();

                updatePedal(100);

            }

        }
    );


    document.addEventListener(
        "keyup",
        function(event) {

            if (
                event.code === "Space"
            ) {

                event.preventDefault();

                updatePedal(0);

            }

        }
    );


    /*
    Initial state
    */

    updatePedal(0);

}
"""


brake_component = st.components.v2.component(
    "ev_brake_pedal",
    html=BRAKE_HTML,
    css=BRAKE_CSS,
    js=BRAKE_JS
)


def brake_pedal(
    default=0.0,
    key="ev_brake_pedal_instance"
):

    result = brake_component(
        key=key,

        default={
            "pressure": float(default)
        },

        on_pressure_change=lambda: None
    )

    if result.pressure is None:
        return float(default)

    return float(result.pressure)