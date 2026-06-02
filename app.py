import streamlit as st
import tempfile
import os
import traceback
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


st.set_page_config(
    page_title="Vehicle Order Analysis Tool",
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


TARGETS = {
    "Diesel": {
        "Front Axle": {
            "rpm": np.array([1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]),
            "amp": np.array([2.5, 2.5, 2.5, 7.5, 7.5, 7.5, 7.5, 7.5])
        },
        "Rear Axle": {
            "rpm": np.array([1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]),
            "amp": np.array([2.5, 2.5, 2.5, 7.5, 7.5, 7.5, 7.5, 7.5])
        }
    },
    "Gasoline": {
        "Front Axle": {
            "rpm": np.array([1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]),
            "amp": np.array([2.5, 2.5, 2.5, 6.25, 10.0, 10.0, 10.0, 10.0])
        },
        "Rear Axle": {
            "rpm": np.array([1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]),
            "amp": np.array([5.0, 5.0, 5.0, 10.0, 12.5, 12.5, 12.5, 12.5])
        }
    }
}


def make_excel_report(vehicle_info, result_df, curve_df):
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame([vehicle_info]).to_excel(
            writer,
            sheet_name="Vehicle Info",
            index=False
        )

        result_df.to_excel(
            writer,
            sheet_name="Target Comparison",
            index=False
        )

        curve_df.to_excel(
            writer,
            sheet_name="Order Curves",
            index=False
        )

    output.seek(0)
    return output


st.subheader("Vehicle Information")

col1, col2, col3 = st.columns(3)

with col1:
    vin_number = st.text_input(
        "VIN Number",
        placeholder="Enter vehicle VIN number"
    )

with col2:
    fuel_type = st.selectbox(
        "Fuel Type",
        ["Select fuel type", "Diesel", "Gasoline"]
    )

with col3:
    axle_type = st.selectbox(
        "Axle Type",
        ["Select axle type", "Front Axle", "Rear Axle"]
    )


st.subheader("Measurement Data")

uploaded_file = st.file_uploader(
    "Upload Excel Data File",
    type=["xlsx"]
)


can_continue = (
    vin_number.strip() != ""
    and fuel_type != "Select fuel type"
    and axle_type != "Select axle type"
    and uploaded_file is not None
)

if not can_continue:
    st.warning("Please enter VIN number, select fuel type, select axle type, and upload Excel file.")
    st.stop()


target_rpm = TARGETS[fuel_type][axle_type]["rpm"]
target_amp = TARGETS[fuel_type][axle_type]["amp"]

st.success("Vehicle information and Excel file are ready for analysis.")

info_cols = st.columns(3)
info_cols[0].metric("VIN", vin_number)
info_cols[1].metric("Fuel Type", fuel_type)
info_cols[2].metric("Axle Type", axle_type)


st.subheader("Analysis Settings")

with st.expander("Advanced Settings", expanded=False):

    selected_channel = st.selectbox(
        "Order Map Channel",
        ["ChA", "ChB", "ChC"]
    )

    samples_per_rev = st.slider(
        "Samples per revolution",
        128,
        2048,
        512
    )

    revs_per_block = st.number_input(
        "Revs per block",
        min_value=2,
        max_value=64,
        value=8,
        step=1
    )

    overlap = st.slider(
        "Overlap",
        min_value=0.0,
        max_value=0.9,
        value=0.75,
        step=0.05
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

    order_width = st.number_input(
        "Order width",
        min_value=0.05,
        max_value=2.0,
        value=0.15,
        step=0.05
    )

    rpm_step = st.number_input(
        "RPM step",
        min_value=1,
        max_value=100,
        value=10,
        step=1
    )

    cal_factor = st.number_input(
        "Amplitude Calibration Factor",
        min_value=0.01,
        max_value=10.0,
        value=1.0,
        step=0.01
    )


if st.button("Run Order Analysis", type="primary"):

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

        with st.spinner("Order analysis is running..."):

            channel_curves = {}
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
                    revs_per_block=revs_per_block,
                    overlap=overlap,
                    max_order=max_order
                )

                rpm_sorted, amp_sorted = extract_order_vs_rpm(
                    orders,
                    rpms,
                    spec,
                    target_order=target_order,
                    width=order_width,
                    rpm_step=rpm_step,
                    smooth=True
                )

                amp_sorted = amp_sorted * cal_factor

                channel_curves[name] = {
                    "rpm": rpm_sorted,
                    "amp": amp_sorted
                }

                peak_idx = np.argmax(amp_sorted)
                peak_rpm = float(rpm_sorted[peak_idx])
                peak_amp = float(amp_sorted[peak_idx])

                target_at_peak = float(
                    np.interp(
                        peak_rpm,
                        target_rpm,
                        target_amp
                    )
                )

                margin = peak_amp - target_at_peak
                margin_percent = (
                    margin / target_at_peak * 100.0
                    if target_at_peak > 0
                    else np.nan
                )

                status = "PASS" if peak_amp <= target_at_peak else "FAIL"

                peak_results.append({
                    "Channel": name,
                    "Peak RPM": peak_rpm,
                    "Peak Amplitude [m/s²]": peak_amp,
                    "Target at Peak RPM [m/s²]": target_at_peak,
                    "Margin [m/s²]": margin,
                    "Margin [%]": margin_percent,
                    "Status": status
                })

            result_df = pd.DataFrame(peak_results)

            overall_status = (
                "PASS"
                if (result_df["Status"] == "PASS").all()
                else "FAIL"
            )

            tab1, tab2, tab3 = st.tabs(
                [
                    "Target Comparison",
                    "Order Map",
                    "Raw Results"
                ]
            )

            with tab1:

                st.subheader("Result Summary")

                kpi1, kpi2, kpi3, kpi4 = st.columns(4)

                kpi1.metric(
                    "Peak ChA",
                    f"{result_df.loc[result_df['Channel'] == 'ChA', 'Peak Amplitude [m/s²]'].iloc[0]:.2f} m/s²"
                )

                kpi2.metric(
                    "Peak ChB",
                    f"{result_df.loc[result_df['Channel'] == 'ChB', 'Peak Amplitude [m/s²]'].iloc[0]:.2f} m/s²"
                )

                kpi3.metric(
                    "Peak ChC",
                    f"{result_df.loc[result_df['Channel'] == 'ChC', 'Peak Amplitude [m/s²]'].iloc[0]:.2f} m/s²"
                )

                kpi4.metric(
                    "Overall Assessment",
                    overall_status
                )

                st.subheader(f"{target_order}. Order vs RPM with Target Curve")

                fig2, ax2 = plt.subplots(figsize=(12, 7))

                for name, curve in channel_curves.items():
                    ax2.plot(
                        curve["rpm"],
                        curve["amp"],
                        label=name
                    )

                ax2.plot(
                    target_rpm,
                    target_amp,
                    color="red",
                    linewidth=4,
                    label="Target Curve"
                )

                ax2.set_xlabel("RPM")
                ax2.set_ylabel(
                    f"{target_order}. Order Amplitude [m/s²] Calibrated"
                )

                ax2.set_title(
                    f"{target_order}. Order vs RPM | VIN: {vin_number} | {fuel_type} | {axle_type}"
                )

                ax2.grid(True, alpha=0.3)
                ax2.legend()

                st.pyplot(fig2)

                st.subheader("Target Compliance")

                st.dataframe(
                    result_df,
                    use_container_width=True
                )

                if overall_status == "PASS":
                    st.success("Overall Assessment: PASS")
                else:
                    st.error("Overall Assessment: FAIL")

                png_buffer = BytesIO()
                fig2.savefig(
                    png_buffer,
                    format="png",
                    dpi=200,
                    bbox_inches="tight"
                )
                png_buffer.seek(0)

                st.download_button(
                    label="Download Target Comparison PNG",
                    data=png_buffer,
                    file_name=f"{vin_number}_target_comparison.png",
                    mime="image/png"
                )

            with tab2:

                st.subheader(f"Order Map - {selected_channel}")

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
                    revs_per_block=revs_per_block,
                    overlap=overlap,
                    max_order=max_order
                )

                idx = np.argsort(rpms)
                r = rpms[idx]
                s = spec[idx]

                db = 20 * np.log10(
                    np.maximum(s * cal_factor, 1e-12)
                )

                fig, ax = plt.subplots(figsize=(12, 7))

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
                    f"Order Map - {selected_channel} | VIN: {vin_number} | {fuel_type} | {axle_type}"
                )

                st.pyplot(fig)

            with tab3:

                st.subheader("Raw Curve Data")

                curve_df = pd.DataFrame()

                base_rpm = None

                for name, curve in channel_curves.items():
                    if base_rpm is None:
                        base_rpm = curve["rpm"]
                        curve_df["RPM"] = base_rpm

                    curve_df[name] = np.interp(
                        base_rpm,
                        curve["rpm"],
                        curve["amp"]
                    )

                curve_df["Target"] = np.interp(
                    curve_df["RPM"],
                    target_rpm,
                    target_amp
                )

                st.dataframe(
                    curve_df,
                    use_container_width=True
                )

                vehicle_info = {
                    "VIN": vin_number,
                    "Fuel Type": fuel_type,
                    "Axle Type": axle_type,
                    "Target Order": target_order,
                    "Order Width": order_width,
                    "RPM Step": rpm_step,
                    "Samples per Rev": samples_per_rev,
                    "Revs per Block": revs_per_block,
                    "Overlap": overlap,
                    "Calibration Factor": cal_factor,
                    "Overall Assessment": overall_status
                }

                excel_report = make_excel_report(
                    vehicle_info,
                    result_df,
                    curve_df
                )

                st.download_button(
                    label="Download Excel Report",
                    data=excel_report,
                    file_name=f"{vin_number}_order_analysis_report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        os.remove(xlsx_path)

    except Exception:
        st.error("Uygulama çalışırken hata oluştu")
        st.code(traceback.format_exc())
