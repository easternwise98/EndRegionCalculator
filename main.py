# app.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Tuple
from enum import Enum
import json
from datetime import datetime
import uuid


# ========================================
# Custom CSS Theme
# ========================================

def load_custom_css():
    theme = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
    }
    
    [data-testid="stAppViewContainer"] {
        background-color: #242424;
    }
    
    [data-testid="stSidebar"] {
        background-color: #2E2E2E;
    }
    
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        background-color: #383838;
        border: 1px solid #484848;
        color: #F2F2F2;
        border-radius: 8px;
    }
    
    .stButton > button {
        background-color: #74B9FF;
        color: #242424;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 1.2rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #5FA8D3;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(116, 185, 255, 0.4);
    }
    
    .stDownloadButton > button {
        background-color: #FFEAA7;
        color: #242424;
        border: none;
        border-radius: 8px;
        font-weight: 600;
    }
    
    .stDownloadButton > button:hover {
        background-color: #FDCB6E;
    }
    
    [data-testid="stMetricValue"] {
        color: #74B9FF;
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    [data-testid="stMetricLabel"] {
        color: #B0B0B0;
        font-size: 0.95rem;
        font-weight: 500;
    }
    
    h1 {
        color: #F2F2F2;
        font-weight: 700;
        font-size: 2rem;
    }
    
    h2 {
        color: #F2F2F2;
        font-weight: 600;
        font-size: 1.5rem;
        margin-top: 1.5rem;
    }
    
    h3 {
        color: #B0B0B0;
        font-weight: 500;
        font-size: 1.1rem;
    }
    
    [data-testid="stDataFrame"] {
        background-color: #333333;
        border-radius: 8px;
    }
    
    .stSuccess {
        background-color: rgba(116, 185, 255, 0.15);
        border-left: 4px solid #74B9FF;
        color: #F2F2F2;
    }
    
    hr {
        border: none;
        border-top: 1px solid #484848;
        margin: 1.5rem 0;
    }
    
    [data-testid="stFileUploader"] {
        background-color: #383838;
        border: 2px dashed #484848;
        border-radius: 8px;
        padding: 1rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #383838;
        border-radius: 8px 8px 0 0;
        color: #B0B0B0;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #74B9FF;
        color: #242424;
    }
    
    .calculation-report {
        background-color: #333333;
        padding: 1.5rem;
        border-radius: 8px;
        font-family: 'Consolas', monospace;
        color: #F2F2F2;
        white-space: pre-wrap;
        line-height: 1.6;
    }
    
    @media (prefers-color-scheme: light) {
        [data-testid="stAppViewContainer"] {
            background-color: #F5F7FA;
        }
        
        [data-testid="stSidebar"] {
            background-color: #FFFFFF;
        }
        
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stSelectbox > div > div > select {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            color: #2D3436;
        }
        
        h1, h2 {
            color: #2D3436;
        }
        
        h3 {
            color: #636E72;
        }
        
        [data-testid="stMetricValue"] {
            color: #0984E3;
        }
        
        [data-testid="stMetricLabel"] {
            color: #636E72;
        }
        
        .stButton > button {
            background-color: #0984E3;
            color: #FFFFFF;
        }
        
        .stButton > button:hover {
            background-color: #74B9FF;
        }
        
        .stDownloadButton > button {
            background-color: #FDCB6E;
            color: #2D3436;
        }
        
        [data-testid="stFileUploader"] {
            background-color: #FFFFFF;
            border: 2px dashed #E2E8F0;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: #FFFFFF;
            color: #636E72;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #0984E3;
            color: #FFFFFF;
        }
        
        .calculation-report {
            background-color: #FFFFFF;
            color: #2D3436;
        }
    }
    </style>
    """
    st.markdown(theme, unsafe_allow_html=True)


# ========================================
# Data Models
# ========================================

class SoilType(Enum):
    IGM_SL = "IGM_S&L"
    IGM_M = "IGM_M"
    S_CLAY = "S_Clay"
    S_SAND = "S_Sand"
    MS = "MS"
    WS = "WS"
    R_PL = "R_PL"
    R_ML = "R_ML"
    R_D = "R_D"
    R_FR = "R_FR"


SOIL_TYPE_KR = {
    "IGM_S&L": "풍화토(모래+실트)",
    "IGM_M": "풍화토(점토질)",
    "S_Clay": "연약점토",
    "S_Sand": "연약모래",
    "MS": "중간모래",
    "WS": "풍화암",
    "R_PL": "연암(편마암)",
    "R_ML": "중간암",
    "R_D": "경암",
    "R_FR": "파쇄암"
}


@dataclass
class PileData:
    """개별 말뚝 데이터"""
    id: str
    name: str
    dia: float
    height: float
    condition: str
    ground_elev: float
    scour_depth: float
    grounded_mode: bool
    soil_types: List[str]
    soil_ths: List[float]


@dataclass
class ProjectData:
    """프로젝트 전체 데이터"""
    project_name: str
    created_at: str
    updated_at: str
    fck: float
    ec: float
    unit_weight: float
    piles: List[PileData]


class Concrete:
    def __init__(self, id: int, name: str, fck: float, ec: float, unit_weight: float):
        self.id = id
        self.name = name
        self.ec = ec
        self.fck = fck
        self.unit_weight = unit_weight


@dataclass
class BoringProperty:
    depth: float
    soil_type: SoilType
    e0_uls: float
    e0_ee: float
    kh_uls: float
    kh_ee: float


class Pile:
    def __init__(self, id: int, name: str, dia: float, height: float):
        self.id = id
        self.name = name
        self.dia = dia
        self.height = height
        self.inertia = np.pi * dia ** 4 / 64
        self.concrete: Optional[Concrete] = None
        self.boring: Optional['Boring'] = None

    def set_concrete(self, concrete: Concrete):
        self.concrete = concrete

    def get_end_region_grounded(self, boring: 'Boring', ground_num: int, mode: bool = False) -> tuple:
        moment_maximum_depth = 0.0
        if mode:
            beta = get_beta_factor(self.dia, boring.properties[ground_num].kh_uls,
                                   boring.properties[ground_num].e0_uls)
            moment_maximum_depth = 1 / beta
        else:
            moment_maximum_depth = 2.0 * self.dia

        moment_zero_elev = moment_maximum_depth + 3 * self.dia
        return -1 * self.dia, max(moment_zero_elev, 18 * 25.4)

    def get_end_region_watered(self, boring: 'Boring') -> List[tuple]:
        moment_max_elev = max(boring.ground_elev, boring.scour_elev) + 2 * self.dia
        moment_zero_elev = moment_max_elev + 3 * self.dia

        top_end_region_start = -1 * self.dia
        top_end_region_end = get_end_region_BDS2024(self.dia, moment_max_elev)
        top_end_region = (top_end_region_start, top_end_region_end)

        bot_end_region_start = min(boring.ground_elev, boring.scour_elev) - self.dia
        bot_end_region_end = max(moment_zero_elev, 18 * 25.4)
        bot_end_region = (bot_end_region_start, bot_end_region_end)

        if top_end_region[1] >= bot_end_region[0]:
            merged_end = max(top_end_region[1], bot_end_region[1])
            return [(top_end_region_start, merged_end), (merged_end, merged_end)]
        else:
            return [top_end_region, bot_end_region]

    def plot_end_region(self, boring: 'Boring', condition: str = "watered",
                        ground_num: int = 0, grounded_mode: bool = False):
        """단부구역 시각화"""

        if condition == "watered":
            regions_raw = self.get_end_region_watered(boring)
            title_text = f"{self.name} (수중부)"
        elif condition == "grounded":
            regions_raw = [self.get_end_region_grounded(boring, ground_num, mode=grounded_mode)]
            method = "Beta Method" if grounded_mode else "2D Method"
            title_text = f"{self.name} (육상부: {method})"
        else:
            raise ValueError("condition must be 'watered' or 'grounded'")

        pile_top = 0.0
        pile_bot = -self.height / 1000.0
        ground_level = -boring.ground_elev / 1000.0
        scour_level = -boring.scour_elev / 1000.0

        if condition == "watered":
            effective_ground_elev = max(boring.ground_elev, boring.scour_elev)
            m_max_depth = effective_ground_elev + 2 * self.dia
            m_zero_depth = m_max_depth + 3 * self.dia
        else:
            if grounded_mode:
                beta = get_beta_factor(self.dia, boring.properties[ground_num].kh_uls,
                                       boring.properties[ground_num].e0_uls)
                m_max_depth = boring.ground_elev + (1 / beta)
            else:
                m_max_depth = boring.ground_elev + 2 * self.dia
            m_zero_depth = m_max_depth + 3 * self.dia

        m_max_level = -m_max_depth / 1000.0
        m_zero_level = -m_zero_depth / 1000.0

        plot_regions = []
        for (start, end) in regions_raw:
            upper = -start / 1000.0
            lower = -end / 1000.0
            if upper < lower:
                upper, lower = lower, upper
            plot_regions.append((upper, lower))

        fig = go.Figure()
        x_range = [-1.5, 1.5]
        region_colors = ['rgba(255,140,140,0.35)', 'rgba(140,180,255,0.35)']

        fig.add_trace(go.Scatter(
            x=[0, 0], y=[pile_top, pile_bot],
            mode='lines', line=dict(color='lightgray', width=25),
            name='말뚝', showlegend=True
        ))

        for i, (upper, lower) in enumerate(plot_regions):
            color = region_colors[i % len(region_colors)]
            region_label = f"구역 {i+1}<br>({upper:.2f}~{lower:.2f}m)"

            fig.add_trace(go.Scatter(
                x=[x_range[0], x_range[1], x_range[1], x_range[0], x_range[0]],
                y=[upper, upper, lower, lower, upper],
                fill='toself', fillcolor=color, line=dict(width=0),
                mode='lines', name=f'단부구역 {i+1}',
                text=region_label, hoverinfo='text', showlegend=True
            ))

        fig.add_trace(go.Scatter(
            x=x_range, y=[ground_level, ground_level],
            mode='lines', line=dict(color='#90EE90', width=2, dash='dash'),
            name=f'지반면 ({ground_level:.2f}m)', showlegend=True
        ))

        if condition == "watered":
            fig.add_trace(go.Scatter(
                x=x_range, y=[scour_level, scour_level],
                mode='lines', line=dict(color='#F4A460', width=2, dash='dashdot'),
                name=f'세굴면 ({scour_level:.2f}m)', showlegend=True
            ))

        fig.add_trace(go.Scatter(
            x=x_range, y=[m_max_level, m_max_level],
            mode='lines', line=dict(color='#87CEEB', width=2, dash='dot'),
            name=f'모멘트 최대 ({m_max_level:.2f}m)', showlegend=True
        ))

        fig.add_trace(go.Scatter(
            x=x_range, y=[m_zero_level, m_zero_level],
            mode='lines', line=dict(color='#DDA0DD', width=2, dash='longdash'),
            name=f'모멘트 0점 ({m_zero_level:.2f}m)', showlegend=True
        ))

        fig.update_layout(
            title=dict(text=title_text, x=0.5, font=dict(size=18, color='white', family='Noto Sans KR')),
            xaxis=dict(title="", showticklabels=False, range=x_range, zeroline=False),
            yaxis=dict(title="말뚝 두부로부터 깊이 (m)", zeroline=True,
                       zerolinecolor='white', zerolinewidth=1, gridcolor='rgba(128,128,128,0.3)'),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02,
                        bgcolor="rgba(0,0,0,0.7)", bordercolor="white", borderwidth=1,
                        font=dict(color='white', family='Noto Sans KR')),
            template="plotly_dark", width=900, height=800, showlegend=True
        )

        return fig, {
            'pile_top': pile_top,
            'pile_bot': pile_bot,
            'ground_level': ground_level,
            'scour_level': scour_level if condition == "watered" else None,
            'm_max_level': m_max_level,
            'm_zero_level': m_zero_level,
            'regions': plot_regions
        }


class Boring:
    _E0_ULS_dict = {
        SoilType.IGM_SL: 100.0, SoilType.IGM_M: 50.0,
        SoilType.S_CLAY: 7.0, SoilType.S_SAND: 16.0,
        SoilType.MS: 18.0, SoilType.WS: 22.0,
        SoilType.R_PL: 600.0, SoilType.R_ML: 900.0,
        SoilType.R_D: 750.0, SoilType.R_FR: 400.0,
    }

    _E0_EE_dict = {
        SoilType.IGM_SL: 886.9, SoilType.IGM_M: 886.9,
        SoilType.S_CLAY: 274.0, SoilType.S_SAND: 387.1,
        SoilType.MS: 327.7, SoilType.WS: 572.0,
        SoilType.R_FR: 2015.4, SoilType.R_PL: 2015.4,
        SoilType.R_ML: 2015.4, SoilType.R_D: 2015.4,
    }

    def __init__(self, id: int, name: str):
        self.id = id
        self.name = name
        self.pile: Optional[Pile] = None
        self.ground_elev = 0.0
        self.scour_depth = 0.0
        self.scour_elev = 0.0
        self.types: List[SoilType] = []
        self.ths: List[float] = []
        self.depths: List[float] = []
        self.properties: List[BoringProperty] = []

    def set_ground_by_pile(self, pile: Pile, types: List[SoilType], ths: List[float],
                           ground_elev: float, scour_depth: Optional[float] = 0.0):
        self.ground_elev = ground_elev
        self.scour_depth = scour_depth
        self.scour_elev = ground_elev + scour_depth
        self.types = types
        self.ths = ths
        self.depths = list(np.cumsum(ths))

        for _type, _depth in zip(self.types, self.depths):
            if _type == SoilType.IGM_M and _depth > 80:
                e0_uls = 100.0
            else:
                e0_uls = self._E0_ULS_dict.get(_type, 0.0)

            e0_ee = self._E0_EE_dict.get(_type, 0.0)
            kh_uls = self.get_kh_uls_fd2018(e0_uls, pile.dia, pile.concrete.ec, pile.inertia)
            kh_ee = self.get_kh_ee_fd2018(e0_uls, pile.dia, pile.concrete.ec, pile.inertia)
            self.properties.append(BoringProperty(_depth, _type, e0_uls, e0_ee, kh_uls, kh_ee))

    @staticmethod
    def get_kh_uls_fd2018(e0_mpa: float, dia_mm: float, ec_mpa: float, inertia_mm4: float) -> float:
        e0_kpa = e0_mpa * 1000.0
        ec_kpa = ec_mpa * 1000.0
        inertia_m4 = inertia_mm4 * 1e-12
        dia_m = dia_mm / 1000.0
        return 1.208 * (4 * e0_kpa) ** 1.1 * (dia_m ** -0.31) * (ec_kpa * inertia_m4) ** -0.103

    @staticmethod
    def get_kh_ee_fd2018(e0_mpa: float, dia_mm: float, ec_mpa: float, inertia_mm4: float) -> float:
        e0_kpa = e0_mpa * 1000.0
        ec_kpa = ec_mpa * 1000.0
        inertia_m4 = inertia_mm4 * 1e-12
        dia_m = dia_mm / 1000.0
        return 1.208 * (8 * e0_kpa) ** 1.1 * (dia_m ** -0.31) * (ec_kpa * inertia_m4) ** -0.103


def get_end_region_BDS2024(d, h) -> float:
    return max(d, h / 6, 18 * 25.4)


def get_beta_factor(d_mm, kh_knm3, e0_mpa):
    i_mm4 = np.pi * d_mm ** 4 / 64
    i_m4 = i_mm4 * 1e-12
    d_m = d_mm / 1000.0
    e0_kpa = e0_mpa * 1000.0
    return np.pow((kh_knm3 * d_m) / (4 * e0_kpa * i_m4), 0.25)


# ========================================
# Calculation Report Generator
# ========================================

def generate_calculation_report(pile_data: PileData, fck: float, ec: float, unit_weight: float,
                                pile: Pile, boring: Boring, results: Dict) -> str:
    """계산서 생성"""

    report = []
    report.append("=" * 80)
    report.append("말뚝 단부구역 계산서")
    report.append("=" * 80)
    report.append("")

    # 1. 프로젝트 정보
    report.append("1. 프로젝트 정보")
    report.append("-" * 80)
    report.append(f"  말뚝 명칭: {pile_data.name}")
    report.append(f"  계산 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    # 2. 입력 데이터
    report.append("2. 입력 데이터")
    report.append("-" * 80)
    report.append("  2.1 콘크리트 물성")
    report.append(f"      설계기준압축강도 (fck): {fck} MPa")
    report.append(f"      탄성계수 (Ec): {ec} MPa")
    report.append(f"      단위중량: {unit_weight} kg/m³")
    report.append("")

    report.append("  2.2 말뚝 제원")
    report.append(f"      직경 (D): {pile_data.dia} mm = {pile_data.dia/1000:.3f} m")
    report.append(f"      길이 (L): {pile_data.height} mm = {pile_data.height/1000:.3f} m")
    report.append(f"      단면2차모멘트 (I): {pile.inertia:.4e} mm⁴ = {pile.inertia*1e-12:.4e} m⁴")
    report.append("")

    report.append("  2.3 지반 조건")
    report.append(f"      조건 구분: {pile_data.condition}")
    report.append(f"      지반고: {pile_data.ground_elev} mm = {pile_data.ground_elev/1000:.3f} m")
    if pile_data.condition == "수중부":
        report.append(f"      세굴깊이: {pile_data.scour_depth} mm = {pile_data.scour_depth/1000:.3f} m")
        report.append(f"      세굴면 표고: {boring.scour_elev} mm = {boring.scour_elev/1000:.3f} m")
    else:
        if pile_data.grounded_mode:
            report.append(f"      계산 방법: Beta Method (가상고정점)")
        else:
            report.append(f"      계산 방법: 2D Method")
    report.append("")

    report.append("  2.4 지층 구성")
    report.append(f"      {'Layer':<8} {'지층종류':<15} {'두께(mm)':<12} {'누적깊이(mm)':<15}")
    report.append(f"      {'-'*60}")
    for i, (soil_type, th, depth) in enumerate(zip(pile_data.soil_types, pile_data.soil_ths, boring.depths)):
        soil_name = f"{soil_type} ({SOIL_TYPE_KR[soil_type]})"
        report.append(f"      {i+1:<8} {soil_name:<15} {th:<12.1f} {depth:<15.1f}")
    report.append("")

    # 3. 지반 물성 계산
    report.append("3. 지반 물성 계산")
    report.append("-" * 80)
    report.append(f"  {'Layer':<8} {'E₀(ULS)':<12} {'E₀(EE)':<12} {'Kh(ULS)':<15} {'Kh(EE)':<15}")
    report.append(f"  {'':8} {'(MPa)':<12} {'(MPa)':<12} {'(kN/m³)':<15} {'(kN/m³)':<15}")
    report.append(f"  {'-'*70}")
    for i, prop in enumerate(boring.properties):
        report.append(f"  {i+1:<8} {prop.e0_uls:<12.1f} {prop.e0_ee:<12.1f} {prop.kh_uls:<15.2f} {prop.kh_ee:<15.2f}")
    report.append("")

    # 4. 단부구역 계산
    report.append("4. 단부구역 계산")
    report.append("-" * 80)

    if pile_data.condition == "수중부":
        report.append("  4.1 수중부 단부구역 산정")
        report.append("")
        effective_ground = max(boring.ground_elev, boring.scour_elev)
        report.append(f"  (1) 유효 지반면: max(지반고, 세굴면) = {effective_ground:.1f} mm")
        report.append("")

        m_max_depth = effective_ground + 2 * pile_data.dia
        report.append(f"  (2) 모멘트 최대점 깊이")
        report.append(f"      = 유효지반면 + 2D")
        report.append(f"      = {effective_ground:.1f} + 2×{pile_data.dia:.1f}")
        report.append(f"      = {m_max_depth:.1f} mm")
        report.append("")

        m_zero_depth = m_max_depth + 3 * pile_data.dia
        report.append(f"  (3) 모멘트 0점 깊이")
        report.append(f"      = 모멘트최대점 + 3D")
        report.append(f"      = {m_max_depth:.1f} + 3×{pile_data.dia:.1f}")
        report.append(f"      = {m_zero_depth:.1f} mm")
        report.append("")

        report.append(f"  (4) 상단 단부구역")
        h_value = m_zero_depth
        top_region = get_end_region_BDS2024(pile_data.dia, h_value)
        report.append(f"      = max(D, H/6, 18in)")
        report.append(f"      = max({pile_data.dia:.1f}, {h_value:.1f}/6, {18*25.4:.1f})")
        report.append(f"      = max({pile_data.dia:.1f}, {h_value/6:.1f}, {18*25.4:.1f})")
        report.append(f"      = {top_region:.1f} mm")
        report.append("")

        report.append(f"  (5) 하단 단부구역")
        bot_start = min(boring.ground_elev, boring.scour_elev)
        bot_end = max(m_zero_depth, 18 * 25.4)
        report.append(f"      시작: min(지반고, 세굴면) = {bot_start:.1f} mm")
        report.append(f"      종료: max(모멘트0점, 18in) = {bot_end:.1f} mm")
        report.append("")

    else:  # 육상부
        report.append("  4.1 육상부 단부구역 산정")
        report.append("")

        if pile_data.grounded_mode:
            report.append(f"  (1) Beta Method 적용")
            beta = get_beta_factor(pile_data.dia, boring.properties[0].kh_uls, boring.properties[0].e0_uls)
            report.append(f"      β = [(Kh×D)/(4×E₀×I)]^0.25")
            report.append(f"      β = {beta:.6f} m⁻¹")
            m_max_depth = boring.ground_elev + (1 / beta)
            report.append(f"      모멘트최대점 = 지반고 + 1/β = {m_max_depth:.1f} mm")
        else:
            report.append(f"  (1) 2D Method 적용")
            m_max_depth = boring.ground_elev + 2 * pile_data.dia
            report.append(f"      모멘트최대점 = 지반고 + 2D = {m_max_depth:.1f} mm")
        report.append("")

        m_zero_depth = m_max_depth + 3 * pile_data.dia
        report.append(f"  (2) 모멘트 0점")
        report.append(f"      = 모멘트최대점 + 3D = {m_zero_depth:.1f} mm")
        report.append("")

        end_region = max(m_max_depth + 3 * pile_data.dia, 18 * 25.4)
        report.append(f"  (3) 단부구역 범위")
        report.append(f"      = max(모멘트최대 + 3D, 18in)")
        report.append(f"      = {end_region:.1f} mm")
        report.append("")

    # 5. 최종 결과
    report.append("5. 계산 결과 요약")
    report.append("-" * 80)
    report.append(f"  말뚝 두부: {results['pile_top']:.3f} m")
    report.append(f"  말뚝 하단: {results['pile_bot']:.3f} m")
    report.append(f"  지반면: {results['ground_level']:.3f} m")
    if results['scour_level'] is not None:
        report.append(f"  세굴면: {results['scour_level']:.3f} m")
    report.append(f"  모멘트 최대점: {results['m_max_level']:.3f} m")
    report.append(f"  모멘트 0점: {results['m_zero_level']:.3f} m")
    report.append("")

    report.append("  단부구역:")
    for i, (upper, lower) in enumerate(results['regions']):
        report.append(f"    구역 {i+1}: {upper:.3f} m ~ {lower:.3f} m (길이: {abs(upper-lower):.3f} m)")
    report.append("")

    report.append("=" * 80)
    report.append("계산서 끝")
    report.append("=" * 80)

    return "\n".join(report)


# ========================================
# File I/O Functions
# ========================================

def save_project_data(project: ProjectData) -> str:
    """프로젝트 데이터를 JSON 문자열로 변환"""
    data_dict = asdict(project)
    return json.dumps(data_dict, indent=2, ensure_ascii=False)


def load_project_data(json_str: str) -> ProjectData:
    """JSON 문자열에서 프로젝트 데이터 복원"""
    data_dict = json.loads(json_str)
    piles_data = [PileData(**pile) for pile in data_dict['piles']]
    data_dict['piles'] = piles_data
    return ProjectData(**data_dict)


# ========================================
# Streamlit App
# ========================================

def main():
    st.set_page_config(
        page_title="말뚝 단부구역 계산",
        layout="wide",
        page_icon="⚙️",
        initial_sidebar_state="expanded"
    )

    load_custom_css()

    # Session State 초기화
    if 'project_data' not in st.session_state:
        st.session_state.project_data = None
    if 'pile_counter' not in st.session_state:
        st.session_state.pile_counter = 0
    if 'adding_pile' not in st.session_state:
        st.session_state.adding_pile = False
    if 'calculation_results' not in st.session_state:
        st.session_state.calculation_results = {}
    if 'loaded_json' not in st.session_state:
        st.session_state.loaded_json = None
    if 'project_loaded' not in st.session_state:
        st.session_state.project_loaded = False

    st.title("⚙️ 말뚝 단부구역 계산")
    st.markdown("##### 지반~단부구역 계산")
    st.markdown("---")

    with st.sidebar:
        st.header("📝 프로젝트 관리")

        # 파일 불러오기
        st.subheader("💾 데이터 관리")
        uploaded_file = st.file_uploader(
            "프로젝트 파일 불러오기 (.json)",
            type=['json'],
            help="이전에 저장한 프로젝트 데이터를 불러옵니다",
            key="file_uploader"
        )

        if uploaded_file is not None:
            try:
                json_content = uploaded_file.read().decode('utf-8')
                st.session_state.loaded_json = json_content
                st.info("📄 JSON 파일이 준비되었습니다. '프로젝트 로드' 버튼을 눌러주세요.")
            except Exception as e:
                st.error(f"파일 읽기 실패: {str(e)}")

        # 프로젝트 로드 버튼
        if st.session_state.loaded_json is not None:
            if st.button("🚀 프로젝트 로드 및 계산", type="primary", width="stretch"):
                try:
                    loaded_project = load_project_data(st.session_state.loaded_json)
                    st.session_state.project_data = loaded_project
                    st.session_state.calculation_results = {}
                    st.session_state.project_loaded = True
                    st.success(f"✓ 프로젝트 로드 완료: {loaded_project.project_name}")
                    st.rerun()
                except Exception as e:
                    st.error(f"프로젝트 로드 실패: {str(e)}")

        # 프로젝트 정보
        st.markdown("---")
        st.subheader("▸ 프로젝트 정보")

        project_name = st.text_input(
            "프로젝트 명칭",
            value=st.session_state.project_data.project_name if st.session_state.project_data else "Untitled Project",
            placeholder="예: 00대교 기초 설계",
            disabled=st.session_state.project_loaded
        )

        # 공통 콘크리트 물성
        st.markdown("---")
        st.subheader("▸ 공통 콘크리트 물성")

        fck = st.number_input(
            "설계기준압축강도 fck (MPa)",
            value=float(st.session_state.project_data.fck) if st.session_state.project_data else 28.0,
            min_value=10.0,
            max_value=100.0,
            disabled=st.session_state.project_loaded
        )
        ec = st.number_input(
            "탄성계수 Ec (MPa)",
            value=float(st.session_state.project_data.ec) if st.session_state.project_data else 29299.0,
            min_value=10000.0,
            max_value=50000.0,
            disabled=st.session_state.project_loaded
        )
        unit_weight = st.number_input(
            "단위중량 (kg/m³)",
            value=float(st.session_state.project_data.unit_weight) if st.session_state.project_data else 2500.0,
            min_value=2000.0,
            max_value=3000.0,
            disabled=st.session_state.project_loaded
        )

        st.markdown("---")

        # 말뚝 관리
        st.subheader("▸ 말뚝 목록")

        if st.session_state.project_data and len(st.session_state.project_data.piles) > 0:
            num_piles = len(st.session_state.project_data.piles)
            st.info(f"📌 등록된 말뚝: {num_piles}개")

            for i, pile in enumerate(st.session_state.project_data.piles):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(f"{i+1}. {pile.name}")
                with col2:
                    if not st.session_state.project_loaded:
                        if st.button("🗑️", key=f"del_{pile.id}_{i}"):
                            st.session_state.project_data.piles.pop(i)
                            st.session_state.project_data.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            if pile.id in st.session_state.calculation_results:
                                del st.session_state.calculation_results[pile.id]
                            st.rerun()
        else:
            st.info("📌 등록된 말뚝이 없습니다")

        if not st.session_state.project_loaded:
            if st.button("➕ 새 말뚝 추가", type="primary", width="stretch"):
                st.session_state.adding_pile = True
                st.session_state.pile_counter += 1
                st.rerun()

        st.markdown("---")

        # 프로젝트 저장
        if st.session_state.project_data and len(st.session_state.project_data.piles) > 0:
            updated_project = ProjectData(
                project_name=project_name,
                created_at=st.session_state.project_data.created_at,
                updated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                fck=fck,
                ec=ec,
                unit_weight=unit_weight,
                piles=st.session_state.project_data.piles
            )

            json_data = save_project_data(updated_project)
            filename = f"{project_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            st.download_button(
                label="💾 프로젝트 저장",
                data=json_data,
                file_name=filename,
                mime="application/json",
                width="stretch"
            )

        # 새 프로젝트 시작 버튼 (로드된 프로젝트가 있을 때만)
        if st.session_state.project_loaded:
            st.markdown("---")
            if st.button("🔄 새 프로젝트 시작", width="stretch"):
                st.session_state.project_data = None
                st.session_state.loaded_json = None
                st.session_state.project_loaded = False
                st.session_state.calculation_results = {}
                st.session_state.adding_pile = False
                st.rerun()

    # Main Content
    if st.session_state.adding_pile and not st.session_state.project_loaded:
        display_pile_form(project_name, fck, ec, unit_weight)

    elif st.session_state.project_data and len(st.session_state.project_data.piles) > 0:
        # 자동 계산 수행 (프로젝트 로드 시)
        if st.session_state.project_loaded and not st.session_state.calculation_results:
            perform_all_calculations(st.session_state.project_data, fck, ec, unit_weight)

        tab_names = [pile.name for pile in st.session_state.project_data.piles] + ["📊 전체 비교", "📄 계산서"]
        tabs = st.tabs(tab_names)

        for i, (tab, pile_data) in enumerate(zip(tabs[:-2], st.session_state.project_data.piles)):
            with tab:
                display_pile_analysis(pile_data, fck, ec, unit_weight, i)

        with tabs[-2]:
            display_comparison_view(st.session_state.project_data, fck, ec, unit_weight)

        with tabs[-1]:
            display_calculation_reports(st.session_state.project_data, fck, ec, unit_weight)

    else:
        st.info("👈 사이드바에서 '새 말뚝 추가' 버튼을 눌러 말뚝을 등록하거나 JSON 파일을 불러오세요")


def perform_all_calculations(project_data: ProjectData, fck: float, ec: float, unit_weight: float):
    """모든 말뚝에 대한 계산 자동 수행"""
    for i, pile_data in enumerate(project_data.piles):
        condition_en = "watered" if pile_data.condition == "수중부" else "grounded"
        soil_type_enums = [SoilType(soil_type) for soil_type in pile_data.soil_types]

        concrete = Concrete(id=1, name="Concrete-1", fck=fck, ec=ec, unit_weight=unit_weight)
        pile = Pile(id=i, name=pile_data.name, dia=pile_data.dia, height=pile_data.height)
        pile.set_concrete(concrete)

        boring = Boring(id=i, name=f"Boring-{pile_data.name}")
        boring.set_ground_by_pile(pile, soil_type_enums, pile_data.soil_ths, pile_data.ground_elev, pile_data.scour_depth)

        if condition_en == "watered":
            fig, results = pile.plot_end_region(boring, condition="watered")
        else:
            fig, results = pile.plot_end_region(boring, condition="grounded", ground_num=0, grounded_mode=pile_data.grounded_mode)

        # 계산 결과 저장
        st.session_state.calculation_results[pile_data.id] = {
            'pile_data': pile_data,
            'pile': pile,
            'boring': boring,
            'results': results,
            'fck': fck,
            'ec': ec,
            'unit_weight': unit_weight,
            'fig': fig
        }


def display_pile_form(project_name: str, fck: float, ec: float, unit_weight: float):
    """새 말뚝 입력 폼"""
    st.subheader("🔧 새 말뚝 등록")

    col1, col2 = st.columns(2)

    with col1:
        pile_name = st.text_input("말뚝 명칭", value=f"P-{st.session_state.pile_counter:02d}")
        pile_dia = st.number_input("직경 (mm)", value=2500.0, min_value=500.0, max_value=5000.0, step=100.0)
        pile_height = st.number_input("길이 (mm)", value=75000.0, min_value=5000.0, max_value=150000.0, step=1000.0)

    with col2:
        condition = st.selectbox("조건 구분", ["수중부", "육상부"])
        ground_elev = st.number_input("지반고 (mm)", value=0.0, min_value=0.0, step=100.0)

        if condition == "수중부":
            scour_depth = st.number_input("세굴깊이 (mm)", value=0.0, min_value=0.0, step=100.0)
            grounded_mode = False
        else:
            scour_depth = 0.0
            grounded_mode = st.checkbox("⚡ Beta Method 사용", value=False)

    st.markdown("---")
    st.subheader("지층 구성")

    num_layers = st.number_input("지층 개수", value=3, min_value=1, max_value=15, step=1)

    soil_types = []
    soil_ths = []
    soil_type_options = [e.value for e in SoilType]

    cols = st.columns(3)
    for i in range(num_layers):
        with cols[i % 3]:
            st.markdown(f"**Layer {i+1}**")
            soil_type = st.selectbox(
                "종류",
                soil_type_options,
                key=f"new_type_{i}",
                format_func=lambda x: f"{x} ({SOIL_TYPE_KR[x]})",
                label_visibility="collapsed"
            )
            soil_types.append(soil_type)

            th = st.number_input(
                "두께 (mm)",
                value=4500.0,
                min_value=100.0,
                step=100.0,
                key=f"new_th_{i}",
                label_visibility="collapsed"
            )
            soil_ths.append(th)

    st.markdown("---")

    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        if st.button("✓ 말뚝 추가", type="primary", width="stretch"):
            new_pile = PileData(
                id=str(uuid.uuid4()),
                name=pile_name,
                dia=pile_dia,
                height=pile_height,
                condition=condition,
                ground_elev=ground_elev,
                scour_depth=scour_depth,
                grounded_mode=grounded_mode,
                soil_types=soil_types,
                soil_ths=soil_ths
            )

            if st.session_state.project_data is None:
                st.session_state.project_data = ProjectData(
                    project_name=project_name,
                    created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    updated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    fck=fck,
                    ec=ec,
                    unit_weight=unit_weight,
                    piles=[new_pile]
                )
            else:
                st.session_state.project_data.piles.append(new_pile)
                st.session_state.project_data.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            st.session_state.adding_pile = False
            st.success(f"✓ '{pile_name}' 말뚝이 추가되었습니다")
            st.rerun()

    with col_btn2:
        if st.button("✕ 취소", width="stretch"):
            st.session_state.adding_pile = False
            st.rerun()


def display_pile_analysis(pile_data: PileData, fck: float, ec: float, unit_weight: float, pile_idx: int):
    """개별 말뚝 계산 결과 표시"""

    # 이미 계산된 결과가 있으면 사용
    if pile_data.id in st.session_state.calculation_results:
        calc_data = st.session_state.calculation_results[pile_data.id]
        results = calc_data['results']
        fig = calc_data['fig']
    else:
        # 계산 수행
        condition_en = "watered" if pile_data.condition == "수중부" else "grounded"
        soil_type_enums = [SoilType(soil_type) for soil_type in pile_data.soil_types]

        concrete = Concrete(id=1, name="Concrete-1", fck=fck, ec=ec, unit_weight=unit_weight)
        pile = Pile(id=pile_idx, name=pile_data.name, dia=pile_data.dia, height=pile_data.height)
        pile.set_concrete(concrete)

        boring = Boring(id=pile_idx, name=f"Boring-{pile_data.name}")
        boring.set_ground_by_pile(pile, soil_type_enums, pile_data.soil_ths, pile_data.ground_elev, pile_data.scour_depth)

        if condition_en == "watered":
            fig, results = pile.plot_end_region(boring, condition="watered")
        else:
            fig, results = pile.plot_end_region(boring, condition="grounded", ground_num=0, grounded_mode=pile_data.grounded_mode)

        # 계산 결과 저장
        st.session_state.calculation_results[pile_data.id] = {
            'pile_data': pile_data,
            'pile': pile,
            'boring': boring,
            'results': results,
            'fck': fck,
            'ec': ec,
            'unit_weight': unit_weight,
            'fig': fig
        }

    st.subheader("📊 주요 결과")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("말뚝 두부", f"{results['pile_top']:.2f} m")
    with col2:
        st.metric("지반면", f"{results['ground_level']:.2f} m")
    with col3:
        st.metric("모멘트 최대점", f"{results['m_max_level']:.2f} m")
    with col4:
        st.metric("모멘트 0점", f"{results['m_zero_level']:.2f} m")

    if results['scour_level'] is not None:
        col5, _, _, _ = st.columns(4)
        with col5:
            st.metric("세굴면", f"{results['scour_level']:.2f} m")

    st.markdown("---")

    st.subheader("📋 단부구역 좌표")
    regions_df = pd.DataFrame([
        {"구역": f"구역 {i+1}", "상단 (m)": f"{upper:.2f}", "하단 (m)": f"{lower:.2f}"}
        for i, (upper, lower) in enumerate(results['regions'])
    ])
    st.dataframe(regions_df, hide_index=True)

    st.markdown("---")

    st.subheader("📈 시각화")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🌍 지층 물성")

    if pile_data.id in st.session_state.calculation_results:
        boring = st.session_state.calculation_results[pile_data.id]['boring']
        soil_df = pd.DataFrame([
            {
                "깊이 (mm)": f"{prop.depth:.0f}",
                "지층": f"{prop.soil_type.value} ({SOIL_TYPE_KR[prop.soil_type.value]})",
                "E₀ᵤₗₛ (MPa)": f"{prop.e0_uls:.1f}",
                "E₀ₑₑ (MPa)": f"{prop.e0_ee:.1f}",
                "Kₕᵤₗₛ": f"{prop.kh_uls:.2f}",
                "Kₕₑₑ": f"{prop.kh_ee:.2f}"
            }
            for prop in boring.properties
        ])
        st.dataframe(soil_df, hide_index=True)


def display_comparison_view(project_data: ProjectData, fck: float, ec: float, unit_weight: float):
    """전체 말뚝 비교 뷰"""
    st.subheader("📊 전체 말뚝 비교")

    comparison_data = []

    for pile_data in project_data.piles:
        if pile_data.id in st.session_state.calculation_results:
            calc_data = st.session_state.calculation_results[pile_data.id]
            results = calc_data['results']

            comparison_data.append({
                "말뚝명": pile_data.name,
                "조건": pile_data.condition,
                "직경 (mm)": f"{pile_data.dia:.0f}",
                "길이 (mm)": f"{pile_data.height:.0f}",
                "지반고 (m)": f"{results['ground_level']:.2f}",
                "M.Max (m)": f"{results['m_max_level']:.2f}",
                "M.Zero (m)": f"{results['m_zero_level']:.2f}",
                "단부구역 수": len(results['regions'])
            })

    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, hide_index=True)
    else:
        st.warning("⚠️ 계산 결과가 없습니다. 각 말뚝 탭을 확인하세요.")

    st.markdown("---")
    st.info("💡 각 말뚝의 상세 결과는 해당 말뚝 탭에서 확인하세요")


def display_calculation_reports(project_data: ProjectData, fck: float, ec: float, unit_weight: float):
    """계산서 표시"""
    st.subheader("📄 계산서")

    if not st.session_state.calculation_results:
        st.warning("⚠️ 계산 결과가 없습니다. 각 말뚝 탭에서 계산을 먼저 수행하세요.")
        return

    # 말뚝 선택
    pile_names = [pile.name for pile in project_data.piles if pile.id in st.session_state.calculation_results]

    if not pile_names:
        st.warning("⚠️ 계산 결과가 없습니다. 각 말뚝 탭에서 계산을 먼저 수행하세요.")
        return

    selected_pile_name = st.selectbox("계산서 확인할 말뚝 선택", pile_names)

    # 선택된 말뚝의 계산 결과 가져오기
    selected_pile_data = None
    for pile in project_data.piles:
        if pile.name == selected_pile_name and pile.id in st.session_state.calculation_results:
            selected_pile_data = pile
            break

    if selected_pile_data:
        calc_data = st.session_state.calculation_results[selected_pile_data.id]

        # 계산서 생성
        report = generate_calculation_report(
            calc_data['pile_data'],
            calc_data['fck'],
            calc_data['ec'],
            calc_data['unit_weight'],
            calc_data['pile'],
            calc_data['boring'],
            calc_data['results']
        )

        # 계산서 표시
        st.markdown('<div class="calculation-report">' + report.replace('\n', '<br>') + '</div>',
                    unsafe_allow_html=True)

        # 다운로드 버튼
        st.download_button(
            label="📥 계산서 다운로드 (TXT)",
            data=report,
            file_name=f"계산서_{selected_pile_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            width="stretch"
        )


if __name__ == "__main__":
    main()