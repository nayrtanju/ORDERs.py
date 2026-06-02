import streamlit as st
import tempfile
import os
import traceback
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(
    page_title="Order Map Analysis",
    layout="wide"
)

st.title("Vehicle Order Analysis Tool")

try:
    from order_analysis import (
        read_xlsx_numeric,
        angular_resample,
        order_map,
        extract_order_vs_rpm
    )
except Exception:
    st.error("order_analysis.py yüklenirken hata oluştu")
    st.code(traceback.format_exc())
    st.stop()


st.subheader("Vehicle Information")

vin_number = st.text_input(
    "VIN Number",
    placeholder="Enter vehicle VIN number"
)

fuel_type = st.selectbox(
    "Fuel Type",
    options=["Select fuel type", "Diesel", "Gasoline"]
)

uploaded_file = st.file_uploader(
    "Upload Excel Data File",
    type=["xlsx"]
)

can_continue = (
    vin_number.strip() != ""
    and fuel_type != "Select fuel type"
    and uploaded_file is not None
)

if not can_continue:
    st.warning("Please enter VIN number, select fuel type, and upload Excel file.")
    st.stop()

st.success("Vehicle information and Excel file are ready for analysis.")

st.write(f"**VIN:** {vin_number}")
st.write(f"**Fuel Type:** {fuel_type}")

try:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        tmp.write(uploaded_file.read())
        xlsx_path = tmp.name

    headers, data = read_xlsx_numeric(xlsx_path)

    time = data[:, 0]
    rpm = data[:, 4]

    channels = {
        "ChA": data[:, 1],
        "ChB": data[:, 2],
        "ChC": data[:, 3],
    }

    st.subheader("Analysis Settings")

    selected_channel = st.selectbox(
        "Order Map Channel",
        list(channels.keys())
    )

    samples_per_rev = st.slider(
        "Samples per revolution",
        128,
        2048,
        512
    )

    max_order = st.slider(
        "Max order",
        5,
        50,
        30
    )

    target_order = st.number_input(
        "Target order",
        value=10.0
    )

    cal_factor = st.number_input(
        "Amplitude Calibration Factor",
        min_value=0.01,
        max_value=10.0,
        value=1.0,
        step=0.01
    )

    if st.button("Run Order Analysis"):
        with st.spinner("Order analysis is running..."):

            sig = channels[selected_channel]

            theta_u, x_u, rpm_u = angular_resample(
                time,
                rpm,
                sig,
                samples_per_rev=samples_per_rev
            )

            orders, rpms, spec = order_map(
                theta_u,
                x_u,
                rpm_u,
                samples_per_rev=samples_per_rev,
                revs_per_block=8,
                overlap=0.75,
                max_order=max_order
            )

            idx = np.argsort(rpms)
            r = rpms[idx]
            s = spec[idx]

            db = 20 * np.log10(np.maximum(s * cal_factor, 1e-12))

            st.subheader(f"Order Map - {selected_channel}")

            fig, ax = plt.subplots(figsize=(11, 7))

            im = ax.imshow(
                db,
                aspect="auto",
                origin="lower",
                extent=[orders[0], orders[-1], r[0], r[-1]],
                interpolation="nearest",
                cmap="jet"
            )

            fig.colorbar(
                im,
                ax=ax,
                label="Amplitude [dB re 1 m/s²]"
            )

            ax.set_xlabel("Order")
            ax.set_ylabel("RPM")
            ax.set_title(
                f"Order Map - {selected_channel} | VIN: {vin_number} | {fuel_type}"
            )

            st.pyplot(fig)

            st.subheader(f"{target_order}. Order vs RPM - All Channels")

            fig2, ax2 = plt.subplots(figsize=(11, 7))

            peak_results = []

            for name, sig in channels.items():

                theta_u, x_u, rpm_u = angular_resample(
                    time,
                    rpm,
                    sig,
                    samples_per_rev=samples_per_rev
                )

                orders, rpms, spec = order_map(
                    theta_u,
                    x_u,
                    rpm_u,
                    samples_per_rev=samples_per_rev,
                    revs_per_block=8,
                    overlap=0.75,
                    max_order=max_order
                )

                rpm_sorted, amp_sorted = extract_order_vs_rpm(
                    orders,
                    rpms,
                    spec,
                    target_order=target_order,
                    width=0.15,
                    rpm_step=10,
                    smooth=True
                )

                amp_sorted = amp_sorted * cal_factor

                ax2.plot(
                    rpm_sorted,
                    amp_sorted,
                    label=name
                )

                peak_idx = np.argmax(amp_sorted)

                peak_results.append({
                    "Channel": name,
                    "Peak RPM": float(rpm_sorted[peak_idx]),
                    "Peak Amplitude [m/s²]": float(amp_sorted[peak_idx])
                })

            ax2.set_xlabel("RPM")
            ax2.set_ylabel(
                f"{target_order}. Order Amplitude [m/s²] Calibrated"
            )
            ax2.set_title(
                f"{target_order}. Order vs RPM | VIN: {vin_number} | {fuel_type} | Cal Factor = {cal_factor}"
            )
            ax2.grid(True, alpha=0.3)
            ax2.legend()

            st.pyplot(fig2)

            st.subheader("Peak Results")

            st.dataframe(
                peak_results,
                use_container_width=True
            )

    os.remove(xlsx_path)

except Exception:
    st.error("Uygulama çalışırken hata oluştu")
    st.code(traceback.format_exc())
