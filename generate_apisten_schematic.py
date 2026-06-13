import os
from graphviz import Digraph

def main():
    # Initialize the core Graphviz canvas
    g = Digraph(
        "APISTEN_Architecture",
        filename="APISTEN_architecture_schematic",
        format="pdf"
    )

    # ------------------------------------------------------------
    # GLOBAL SCHEMATIC STYLING (RSE/IEEE STANDARDS)
    # ------------------------------------------------------------
    g.attr(
        rankdir="LR",       # Enforce uniform Left-to-Right layout flow
        splines="ortho",    # Clean, right-angled engineering connection routes
        nodesep="0.4",      # Vertical separation spacing between parallel layers
        ranksep="0.8",      # Horizontal spacing between sequential stages
        pad="0.3"
    )

    # Default node visualization geometry
    g.attr(
        "node",
        shape="box",
        style="filled,rounded",
        fillcolor="#F8F9FA", # Neutral off-white canvas
        color="#6C757D",     # Charcoal boundaries
        fontsize="11",
        fontname="Helvetica",
        margin="0.12,0.08"
    )

    # Default edge connection geometry
    g.attr(
        "edge",
        color="#495057",
        fontsize="9",
        fontname="Helvetica",
        arrowsize="0.7"
    )

    # ============================================================
    # STAGE 1: MULTI-DIMENSIONAL DATA INGESTION
    # ============================================================
    with g.subgraph(name="cluster_stage1") as s1:
        s1.attr(
            label="STAGE 1: INPUT FEATURE SPACE",
            fontsize="13",
            fontname="Helvetica-Bold",
            style="filled",
            fillcolor="#E9ECEF", # Soft slate gray background
            color="#CED4DA"
        )

        s1.node(
            "bands",
            "Channels 1–6\nNative Surface Reflectance\n(B02, B03, B04, B05, B06, B07)"
        )

        s1.node(
            "ratios",
            "Channels 7–10\nBio-Optical Feature Ratios\n"
            "r1 = B02/B03\nr2 = B03/B04\n"
            "r3 = B03/(B02+B04)\nr4 = NDCI"
        )

        s1.node(
            "doy",
            "Temporal Index\nDay-of-Year (DOY)"
        )

        s1.node(
            "harmonic",
            "Harmonic Parameterization\n"
            "t_mod = 0.2 + 0.9 * [1 + cos(θ)] / 2\n"
            "where θ = 2π·DOY / 365.25"
        )

        s1.node(
            "temporal_sheet",
            "Channel 11\nContinuous Seasonal Embedding\n(Uniform 2D Spatial Sheet)"
        )

        s1.node(
            "tensor11",
            "Unified Input Tensor X\n(H x W x 11 Channels)",
            shape="box3d",
            fillcolor="#E2EBF1",
            color="#2B5B84"
        )

        s1.edge("doy", "harmonic")
        s1.edge("harmonic", "temporal_sheet")
        s1.edge("bands", "tensor11")
        s1.edge("ratios", "tensor11")
        s1.edge("temporal_sheet", "tensor11")

    # ============================================================
    # STAGE 2: MULTI-MEMBER SPATIO-TEMPORAL ENSEMBLE
    # ============================================================
    with g.subgraph(name="cluster_stage2") as s2:
        s2.attr(
            label="STAGE 2: SPATIO-TEMPORAL CONVOLUTIONAL ENSEMBLE (M = 5)",
            fontsize="13",
            fontname="Helvetica-Bold",
            style="filled",
            fillcolor="#E8F0FE", # Hydro blue thematic background shade
            color="#B4CDFA"
        )

        s2.node(
            "ensemble_input",
            "Parallel Broadcast Layer\n(Shared Data Channels)",
            fillcolor="#FFFFFF"
        )

        # Detailed expanded view of convolutional Member 1
        s2.node("encoder", "Member 1:\nContracting Encoder Arm", fillcolor="#FFFFFF")
        s2.node("skip", "Horizontal Latent\nSkip Connections", style="filled,dashed", fillcolor="#FFFFFF")
        s2.node("decoder", "Expanding Decoder Arm\n(Resolution Recovery)", fillcolor="#FFFFFF")
        s2.node("yhat1", "Output Field: Ŷ¹", fillcolor="#FFFFFF")

        # Independent tracking layers
        s2.node("m2", "Member 2\n(Stochastic U-Net)", fillcolor="#FFFFFF")
        s2.node("m3", "Member 3\n(Stochastic U-Net)", fillcolor="#FFFFFF")
        s2.node("m4", "Member 4\n(Stochastic U-Net)", fillcolor="#FFFFFF")
        s2.node("m5", "Member 5\n(Stochastic U-Net)", fillcolor="#FFFFFF")

        s2.node("yhat2", "Ŷ²", fillcolor="#FFFFFF")
        s2.node("yhat3", "Ŷ³", fillcolor="#FFFFFF")
        s2.node("yhat4", "Ŷ⁴", fillcolor="#FFFFFF")
        s2.node("yhat5", "Ŷ⁵", fillcolor="#FFFFFF")

        s2.edge("ensemble_input", "encoder")
        s2.edge("encoder", "decoder")
        s2.edge("encoder", "skip", style="dashed")
        s2.edge("skip", "decoder", style="dashed")
        s2.edge("decoder", "yhat1")

        s2.edge("ensemble_input", "m2")
        s2.edge("ensemble_input", "m3")
        s2.edge("ensemble_input", "m4")
        s2.edge("ensemble_input", "m5")

        s2.edge("m2", "yhat2")
        s2.edge("m3", "yhat3")
        s2.edge("m4", "yhat4")
        s2.edge("m5", "yhat5")

    # ============================================================
    # STAGE 3: PARALLEL BOUNDED TRUST-REGION GATING
    # ============================================================
    with g.subgraph(name="cluster_stage3") as s3:
        s3.attr(
            label="STAGE 3: ADAPTIVE TRUST-REGION SCALING",
            fontsize="13",
            fontname="Helvetica-Bold",
            style="filled",
            fillcolor="#E6F4EA", # Biophysical green background shade
            color="#A8DAB5"
        )

        s3.node(
            "ratio_branch",
            "Diagnostic Extraction Track\nOverpass Ratio (B02 / B03)",
            fillcolor="#FFFFFF"
        )

        s3.node(
            "anchor",
            "Frozen Seasonal Anchor Profiles\nBaseline Mean (μ) & Variance (σ)",
            fillcolor="#FFFFFF"
        )

        s3.node(
            "zscore",
            "Out-of-Sample Standardization\nZ = (Ratio_overpass − μ) / σ",
            fillcolor="#FFFFFF"
        )

        s3.node(
            "gaussian",
            "Truncated Gaussian Umbrella\n"
            "λ_applied = λ_min + (λ_max − λ_min)·e^(-Z²/2)\n"
            "Boundaries: [λ_min = 0.1 , λ_max = 1.0]",
            fillcolor="#FFFFFF",
            color="#137333"
        )

        s3.node(
            "regularizer",
            "Adaptive Physics-Informed\nLoss Regularization Weight",
            fillcolor="#FFFFFF"
        )

        s3.edge("ratio_branch", "zscore")
        s3.edge("anchor", "zscore")
        s3.edge("zscore", "gaussian")
        s3.edge("gaussian", "regularizer")

    # ============================================================
    # STAGE 4: MULTI-MODEL SYNTHESIS & PHYSICAL GUARDRAILS
    # ============================================================
    with g.subgraph(name="cluster_stage4") as s4:
        s4.attr(
            label="STAGE 4: SYNTHESIS & REGULARIZATION FLOORS",
            fontsize="13",
            fontname="Helvetica-Bold",
            style="filled",
            fillcolor="#FFF3CD", # Warning yellow alert tone background
            color="#FFE69C"
        )

        s4.node(
            "voting",
            "Multi-Model Voting Matrix\n(Ensemble Integration Layer)"
        )

        s4.node(
            "sigma",
            "Epistemic Uncertainty Flag\nEnsemble Standard Deviation (σ)",
            fillcolor="#FFFFFF"
        )

        s4.node(
            "yraw",
            "Predicted Spatial Mean Field\nUnconstrained Estimate (Y_raw)",
            fillcolor="#FFFFFF"
        )

        s4.node(
            "guardrail",
            "Positivity Guardrail Floor\nY_final = max(Y_raw, ε)\nwhere ε = 0.01 mg/m³",
            color="#D93025",
            fillcolor="#FCE8E6" # Highlighted red constraint box
        )

        s4.node(
            "products",
            "Exported Multi-Parameter Maps\n(Chl-a , CDOM , TUR)",
            shape="folder",
            fillcolor="#FFF0F6",
            color="#D11A5B"
        )

        s4.edge("voting", "sigma")
        s4.edge("voting", "yraw")
        s4.edge("yraw", "guardrail")
        s4.edge("guardrail", "products")

    # ============================================================
    # SYSTEM INTER-STAGE PIPELINES & CRITICAL FILTER CUTOUTS
    # ============================================================
    # Core tensor link
    g.edge("tensor11", "ensemble_input")

    # Connect the Stage 1 spectral index directly to the parallel Stage 3 sub-track
    g.edge(
        "ratios", 
        "ratio_branch", 
        label="Extract Blue/Green Ratio",
        color="#1A73E8",
        style="bold"
    )

    # CRITICAL BUG FIX: Added constraint="false" to preserve the clean horizontal ranks
    g.edge(
        "regularizer", 
        "encoder", 
        style="dashed", 
        label="Pixel-wise λ_applied Constraint",
        color="#137333",
        constraint="false"
    )

    # Route all 5 individual member predictions directly into the synthesis pool
    for output_track in ["yhat1", "yhat2", "yhat3", "yhat4", "yhat5"]:
        g.edge(output_track, "voting")

    # Execute compilation
    g.render(cleanup=True)
    print("Execution Complete: Vector 'APISTEN_architecture_schematic.svg' successfully built.")

if __name__ == "__main__":
    main()