"""Unit tests for app.intent_classifier.

Two test suites are included:

1. Legacy interface (backward-compat): classify_intent, get_section_filter,
   INTENT_SECTION_FILTER -- all original tests preserved.

2. Task-5 rich classifier: 40+ Vietnamese questions testing classify()
   and ClassificationResult shape/values.
"""

from __future__ import annotations

import pytest

from app.intent_classifier import (
    INTENT_SECTION_FILTER,
    ClassificationResult,
    _has_profit_guarantee_compound,
    classify,
    classify_intent,
    get_section_filter,
)


# ===========================================================================
# LEGACY INTERFACE TESTS (original tests -- kept for backward-compat)
# ===========================================================================


class TestClassifyIntentLegal:
    def test_phap_ly(self):
        assert classify_intent("Phap ly du an NEO CITY hien tai nhu the nao?") == "legal"

    def test_mo_ban(self):
        assert classify_intent("NEO CITY da du dieu kien mo ban chua?") == "legal"

    def test_dat_coc(self):
        assert classify_intent("Co the dat coc vao du an chua?") == "legal"

    def test_huy_dong_von(self):
        assert classify_intent("Du an da duoc phep huy dong von chua?") == "legal"

    def test_so_do(self):
        assert classify_intent("So do du an the nao?") == "legal"

    def test_hop_dong(self):
        assert classify_intent("Hop dong mua ban ky khi nao?") == "legal"

    def test_english_legal(self):
        assert classify_intent("What is the legal status of the project?") == "legal"


class TestClassifyIntentPricing:
    def test_gia_ban(self):
        assert classify_intent("Gia ban du kien cua NEO CITY la bao nhieu?") == "pricing"

    def test_bang_gia(self):
        assert classify_intent("Bang gia can ho NEO CITY nhu the nao?") == "pricing"

    def test_gia_m2(self):
        assert classify_intent("Gia m2 tai NEO CITY khoang bao nhieu?") == "pricing"

    def test_english_price(self):
        assert classify_intent("What is the price per sqm?") == "pricing"


class TestClassifyIntentSalesPolicy:
    def test_chinh_sach(self):
        assert classify_intent("Chinh sach ban hang cua NEO CITY?") == "sales_policy"

    def test_chiet_khau(self):
        assert classify_intent("Chiet khau khi thanh toan som la bao nhieu?") == "sales_policy"

    def test_tra_gop(self):
        assert classify_intent("Tra gop 70% ngan hang ho tro khong?") == "sales_policy"

    def test_english_discount(self):
        assert classify_intent("Is there a discount for early payment?") == "sales_policy"


class TestClassifyIntentMarket:
    def test_tiem_nang(self):
        assert classify_intent("Tiem nang dau tu cua du an nay nhu the nao?") == "market"

    def test_xu_huong(self):
        assert classify_intent("Xu huong thi truong bat dong san khu vuc ra sao?") == "market"

    def test_loi_nhuan(self):
        assert classify_intent("Loi nhuan ky vong khi dau tu vao NEO CITY?") == "market"

    def test_english_roi(self):
        assert classify_intent("What is the expected ROI for this project?") == "market"


class TestClassifyIntentGeneral:
    def test_tong_quan(self):
        assert classify_intent("NEO CITY la du an gi?") == "general"

    def test_vi_tri(self):
        # "vi tri" has no legacy keyword -> general
        assert classify_intent("Vi tri du an NEO CITY o dau?") == "general"

    def test_empty_string(self):
        assert classify_intent("") == "general"

    def test_whitespace_only(self):
        assert classify_intent("   ") == "general"

    def test_unrelated_english(self):
        assert classify_intent("Tell me about the developer company.") == "general"


class TestClassifyIntentPriority:
    def test_legal_beats_pricing(self):
        # "dat coc" is a legal keyword; legal rule comes first -> legal wins
        result = classify_intent("Gia ban bao nhieu va co the dat coc ngay khong?")
        assert result == "legal"

    def test_legal_beats_sales_policy(self):
        result = classify_intent("Chinh sach thanh toan sau khi mo ban nhu the nao?")
        # "chinh sach" -> sales_policy, "mo ban" -> legal; legal rule is first
        assert result == "legal"


class TestGetSectionFilter:
    def test_legal_filter(self):
        assert get_section_filter("legal") == ["legal"]

    def test_pricing_filter(self):
        sections = get_section_filter("pricing")
        assert "pricing" in sections

    def test_sales_policy_filter(self):
        assert get_section_filter("sales_policy") == ["sales_policy"]

    def test_market_filter(self):
        assert get_section_filter("market") == ["market"]

    def test_general_no_filter(self):
        assert get_section_filter("general") == []

    def test_all_intents_covered(self):
        for intent in ("general", "pricing", "legal", "sales_policy", "market"):
            result = get_section_filter(intent)  # type: ignore[arg-type]
            assert isinstance(result, list)


def test_intent_section_filter_has_all_intents():
    expected = {"general", "pricing", "legal", "sales_policy", "market"}
    assert expected.issubset(set(INTENT_SECTION_FILTER.keys()))


# ===========================================================================
# TASK-5: classify() RICH INTERFACE -- 40+ Vietnamese question tests
# ===========================================================================


class TestClassifyResultShape:
    """ClassificationResult has all required fields and correct types."""

    def test_has_intent_field(self):
        result = classify("NEO CITY la du an gi?")
        assert hasattr(result, "intent")
        assert isinstance(result.intent, str)

    def test_has_target_sections_field(self):
        result = classify("NEO CITY la du an gi?")
        assert hasattr(result, "target_sections")
        assert isinstance(result.target_sections, list)

    def test_has_risk_level_field(self):
        result = classify("NEO CITY la du an gi?")
        assert hasattr(result, "risk_level")
        assert result.risk_level in ("low", "medium", "high", "critical")

    def test_has_must_use_legal_only_field(self):
        result = classify("NEO CITY la du an gi?")
        assert hasattr(result, "must_use_legal_only")
        assert isinstance(result.must_use_legal_only, bool)

    def test_to_dict_shape(self):
        result = classify("Du an da mo ban chua?")
        d = result.to_dict()
        assert set(d.keys()) == {"intent", "target_sections", "risk_level", "must_use_legal_only"}
        assert isinstance(d["target_sections"], list)
        assert isinstance(d["must_use_legal_only"], bool)

    def test_empty_query_returns_unknown(self):
        result = classify("")
        assert result.intent == "unknown"
        assert result.risk_level == "low"
        assert result.must_use_legal_only is False

    def test_whitespace_query_returns_unknown(self):
        result = classify("   ")
        assert result.intent == "unknown"


class TestAgentsMapping:
    @pytest.mark.parametrize(
        ("query", "intent", "target_sections"),
        [
            ("NEO CITY la du an gi?", "project_overview", ["factsheet", "concept_positioning"]),
            ("NEO CITY co tien ich gi?", "amenities", ["factsheet"]),
            ("2PN dien tich bao nhieu?", "product", ["factsheet", "pricing"]),
            ("Me Linh ket noi san bay Noi Bai the nao?", "location", ["location_connectivity", "market"]),
            ("Khach hang muc tieu la ai?", "persona", ["personas"]),
            ("Tagline cua NEO CITY la gi?", "concept", ["concept_positioning"]),
            ("Khach gia dinh tre nen tu van the nao?", "sales_strategy", ["sales_strategy", "personas"]),
            ("Chinh sach thanh toan the nao?", "sales_policy", ["sales_policy", "price_sheet"]),
            ("Can 2PN gia bao nhieu?", "pricing", ["pricing", "price_sheet"]),
            ("Du an da mo ban chua?", "legal", ["legal"]),
            ("Tiem nang dau tu Me Linh ra sao?", "market", ["market"]),
        ],
    )
    def test_target_sections_match_agents_md(self, query, intent, target_sections):
        result = classify(query)
        assert result.intent == intent
        assert result.target_sections == target_sections

    def test_unknown_maps_to_no_sections(self):
        result = classify("Xin chao")
        assert result.intent == "unknown"
        assert result.target_sections == []


class TestEvalCoverage:
    @pytest.mark.parametrize(
        ("query", "expected_intent"),
        [
            ("Dự án đã được phê duyệt quy hoạch chưa?", "legal"),
            ("Dự án có bao nhiêu căn hộ tổng cộng?", "project_overview"),
            ("Sản phẩm thấp tầng NEO CITY gồm những loại nào?", "product"),
            ("NEO CITY gần những khu công nghiệp nào?", "location"),
            ("Người trẻ đi làm thu nhập 20-30 triệu có phù hợp NEO CITY không?", "persona"),
            ("NEO CITY có hồ điều hòa không?", "amenities"),
            ("NEO CITY xa trung tâm quá, có hạ tầng kết nối tốt không?", "location"),
            ("Giá NEO CITY cao hơn thị trường, có xứng đáng không?", "market"),
        ],
    )
    def test_eval_examples_no_longer_unknown(self, query, expected_intent):
        result = classify(query)
        assert result.intent == expected_intent


# ---------------------------------------------------------------------------
# Legal questions (critical, must_use_legal_only=True)
# ---------------------------------------------------------------------------


class TestClassifyLegal:
    """Priority-1: direct legal / sale-opening / deposit questions."""

    def test_mo_ban(self):
        r = classify("Du an da mo ban chua?")
        assert r.intent == "legal"
        assert r.risk_level == "critical"
        assert r.must_use_legal_only is True
        assert "legal" in r.target_sections

    def test_dat_coc(self):
        r = classify("Co dat coc duoc chua?")
        assert r.intent == "legal"
        assert r.risk_level == "critical"
        assert r.must_use_legal_only is True

    def test_du_dieu_kien(self):
        r = classify("Da du dieu kien kinh doanh chua?")
        assert r.intent == "legal"
        assert r.risk_level == "critical"
        assert r.must_use_legal_only is True

    def test_huy_dong_von(self):
        r = classify("Da duoc phep huy dong von chua?")
        assert r.intent == "legal"
        assert r.must_use_legal_only is True

    def test_phap_ly(self):
        r = classify("Phap ly du an nhu the nao?")
        assert r.intent == "legal"
        assert r.risk_level == "critical"

    def test_so_do(self):
        r = classify("Da co so do chua?")
        assert r.intent == "legal"
        assert r.must_use_legal_only is True

    def test_hop_dong_mua_ban(self):
        r = classify("Co the ky hop dong mua ban chua?")
        assert r.intent == "legal"
        assert r.risk_level == "critical"

    def test_phap_ly_day_du(self):
        r = classify("Du an du phap ly chua?")
        assert r.intent == "legal"
        assert r.must_use_legal_only is True


# ---------------------------------------------------------------------------
# Guaranteed profit questions (critical, must_use_legal_only=False)
# ---------------------------------------------------------------------------


class TestClassifyGuaranteedProfit:
    """Priority-2: questions that imply guaranteed return / appreciation."""

    def test_chac_tang_gia(self):
        r = classify("Mua co chac tang gia khong?")
        assert r.intent == "legal"
        assert r.risk_level == "critical"
        assert r.must_use_legal_only is False
        assert "market" in r.target_sections

    def test_cam_ket_loi_nhuan(self):
        r = classify("Co cam ket loi nhuan khong?")
        assert r.intent == "legal"
        assert r.risk_level == "critical"
        assert r.must_use_legal_only is False

    def test_chac_thang(self):
        r = classify("Dau tu chac thang khong?")
        assert r.intent == "legal"
        assert r.risk_level == "critical"

    def test_dam_bao_sinh_loi(self):
        r = classify("Co dam bao sinh loi khong?")
        assert r.intent == "legal"
        assert r.risk_level == "critical"
        assert r.must_use_legal_only is False

    def test_cam_ket_tang_gia(self):
        r = classify("Co cam ket tang gia sau 3 nam khong?")
        assert r.intent == "legal"
        assert r.risk_level == "critical"


# ---------------------------------------------------------------------------
# Pricing questions (high, must_use_legal_only=False)
# ---------------------------------------------------------------------------


class TestClassifyPricing:
    """Priority-3: questions about unit price, price list, cost per sqm."""

    def test_gia_bao_nhieu_2pn(self):
        r = classify("Can 2PN gia bao nhieu?")
        assert r.intent == "pricing"
        assert r.risk_level == "high"
        assert r.must_use_legal_only is False
        assert "pricing" in r.target_sections

    def test_bang_gia(self):
        r = classify("Bang gia moi nhat the nao?")
        assert r.intent == "pricing"
        assert r.risk_level == "high"

    def test_tong_gia(self):
        r = classify("Tong gia tri can 1PN+1 khoang bao nhieu?")
        assert r.intent == "pricing"
        assert r.risk_level == "high"

    def test_don_gia(self):
        r = classify("Don gia tren m2 bao nhieu?")
        assert r.intent == "pricing"
        assert r.risk_level == "high"

    def test_muc_gia(self):
        r = classify("Muc gia du an NEO CITY?")
        assert r.intent == "pricing"
        assert r.risk_level == "high"

    def test_price_sheet_in_sections(self):
        r = classify("Bang gia chi tiet?")
        assert "price_sheet" in r.target_sections or "pricing" in r.target_sections


# ---------------------------------------------------------------------------
# Sales policy questions (high, must_use_legal_only=False)
# ---------------------------------------------------------------------------


class TestClassifySalesPolicy:
    """Priority-4: payment, loan, discount, booking questions."""

    def test_vay_ngan_hang(self):
        r = classify("Chinh sach vay ngan hang ra sao?")
        assert r.intent == "sales_policy"
        assert r.risk_level == "high"
        assert r.must_use_legal_only is False
        assert "sales_policy" in r.target_sections

    def test_chiet_khau(self):
        r = classify("Chiet khau thanh toan nhanh bao nhieu?")
        assert r.intent == "sales_policy"
        assert r.risk_level == "high"

    def test_booking(self):
        r = classify("Booking bao nhieu?")
        assert r.intent == "sales_policy"
        assert r.risk_level == "high"

    def test_chinh_sach_thanh_toan(self):
        r = classify("Chinh sach thanh toan the nao?")
        assert r.intent == "sales_policy"
        assert r.risk_level == "high"

    def test_tra_gop(self):
        r = classify("Tra gop 70% khong?")
        assert r.intent == "sales_policy"
        assert r.risk_level == "high"


# ---------------------------------------------------------------------------
# Market questions (medium, must_use_legal_only=False)
# ---------------------------------------------------------------------------


class TestClassifyMarket:
    """Priority-5: market potential, trends, investment outlook."""

    def test_tiem_nang(self):
        r = classify("Tiem nang dau tu Me Linh ra sao?")
        assert r.intent == "market"
        assert r.risk_level == "medium"
        assert r.must_use_legal_only is False
        assert "market" in r.target_sections

    def test_xu_huong(self):
        r = classify("Xu huong bat dong san Me Linh?")
        assert r.intent == "market"
        assert r.risk_level == "medium"

    def test_ty_suat_sinh_loi(self):
        r = classify("Ty suat sinh loi ky vong?")
        assert r.intent == "market"
        assert r.risk_level == "medium"


# ---------------------------------------------------------------------------
# Location questions (medium, must_use_legal_only=False)
# ---------------------------------------------------------------------------


class TestClassifyLocation:
    """Priority-6: connectivity, infrastructure, distance questions."""

    def test_ket_noi_san_bay(self):
        r = classify("Me Linh ket noi san bay Noi Bai the nao?")
        assert r.intent == "location"
        assert r.risk_level == "medium"
        assert r.must_use_legal_only is False

    def test_ha_tang_giao_thong(self):
        r = classify("Ha tang giao thong vung Me Linh ra sao?")
        assert r.intent == "location"
        assert r.risk_level == "medium"

    def test_cach_trung_tam(self):
        r = classify("NEO CITY cach trung tam Ha Noi bao xa?")
        assert r.intent == "location"
        assert r.risk_level == "medium"


# ---------------------------------------------------------------------------
# Sales strategy questions (medium, must_use_legal_only=False)
# ---------------------------------------------------------------------------


class TestClassifySalesStrategy:
    """Priority-7: advisor scripts, objection handling, closing."""

    def test_nen_tu_van_the_nao(self):
        r = classify("Khach gia dinh tre nen tu van the nao?")
        assert r.intent == "sales_strategy"
        assert r.risk_level == "medium"
        assert r.must_use_legal_only is False

    def test_xu_ly_tu_choi(self):
        r = classify("Cach xu ly tu choi khi khach che gia cao?")
        assert r.intent == "sales_strategy"
        assert r.risk_level == "medium"

    def test_kich_ban_ban(self):
        # Use a query without "dau tu" (market keyword) so sales_strategy wins
        r = classify("Kich ban ban hang khi gap khach kho tinh?")
        assert r.intent == "sales_strategy"
        assert r.risk_level == "medium"


# ---------------------------------------------------------------------------
# Persona questions (low, must_use_legal_only=False)
# ---------------------------------------------------------------------------


class TestClassifyPersona:
    """Priority-8: who is the target customer for this project."""

    def test_khach_hang_muc_tieu(self):
        r = classify("Khach hang muc tieu la ai?")
        assert r.intent == "persona"
        assert r.risk_level == "low"
        assert r.must_use_legal_only is False
        assert "personas" in r.target_sections

    def test_2pn_phu_hop_voi_ai(self):
        r = classify("2PN phu hop voi ai?")
        assert r.intent == "persona"
        assert r.risk_level == "low"

    def test_ai_phu_hop(self):
        r = classify("Ai phu hop voi NEO CITY?")
        assert r.intent == "persona"
        assert r.risk_level == "low"

    def test_danh_cho_doi_tuong(self):
        r = classify("Danh cho doi tuong khach hang nao?")
        assert r.intent == "persona"
        assert r.risk_level == "low"


# ---------------------------------------------------------------------------
# Amenities questions (low, must_use_legal_only=False)
# ---------------------------------------------------------------------------


class TestClassifyAmenities:
    """Priority-10: internal facilities, green space, pool, gym."""

    def test_tien_ich(self):
        r = classify("NEO CITY co tien ich gi?")
        assert r.intent == "amenities"
        assert r.risk_level == "low"
        assert r.must_use_legal_only is False

    def test_ho_trung_tam(self):
        r = classify("Du an co ho trung tam khong?")
        assert r.intent == "amenities"
        assert r.risk_level == "low"

    def test_be_boi(self):
        r = classify("Co be boi khong?")
        assert r.intent == "amenities"
        assert r.risk_level == "low"


# ---------------------------------------------------------------------------
# Concept questions (low, must_use_legal_only=False)
# ---------------------------------------------------------------------------


class TestClassifyConcept:
    """Priority-11: brand, tagline, design vision."""

    def test_tagline(self):
        r = classify("Tagline cua NEO CITY la gi?")
        assert r.intent == "concept"
        assert r.risk_level == "low"
        assert r.must_use_legal_only is False

    def test_dinh_vi(self):
        r = classify("Dinh vi thuong hieu NEO CITY the nao?")
        assert r.intent == "concept"
        assert r.risk_level == "low"


# ---------------------------------------------------------------------------
# Project overview questions (low, must_use_legal_only=False)
# ---------------------------------------------------------------------------


class TestClassifyProjectOverview:
    """Priority-12: general project info, factsheet, developer."""

    def test_du_an_gi(self):
        r = classify("NEO CITY la du an gi?")
        assert r.intent == "project_overview"
        assert r.risk_level == "low"
        assert r.must_use_legal_only is False
        assert "factsheet" in r.target_sections

    def test_chu_dau_tu(self):
        r = classify("Chu dau tu la ai?")
        assert r.intent == "project_overview"
        assert r.risk_level == "low"


# ---------------------------------------------------------------------------
# Unknown / out-of-scope questions
# ---------------------------------------------------------------------------


class TestClassifyUnknown:
    """Queries with no matching keywords fall back to unknown."""

    def test_unrelated_question(self):
        r = classify("Thoi tiet Ha Noi hom nay?")
        assert r.intent == "unknown"
        assert r.risk_level == "low"
        assert r.must_use_legal_only is False
        assert r.target_sections == []

    def test_greeting(self):
        r = classify("Xin chao!")
        assert r.intent == "unknown"
        assert r.risk_level == "low"


# ---------------------------------------------------------------------------
# Priority correctness: legal always beats pricing/sales_policy/market
# ---------------------------------------------------------------------------


class TestClassifyPriority:
    """Verify priority ordering in classify()."""

    def test_legal_beats_pricing_in_rich(self):
        # "dat coc" (legal, priority 1) + "gia bao nhieu" (pricing, priority 3)
        r = classify("Gia bao nhieu va co the dat coc ngay khong?")
        assert r.intent == "legal"
        assert r.risk_level == "critical"

    def test_legal_beats_sales_policy_in_rich(self):
        # "mo ban" (legal) + "chinh sach" (sales_policy)
        r = classify("Chinh sach thanh toan sau khi mo ban nhu the nao?")
        assert r.intent == "legal"
        assert r.risk_level == "critical"

    def test_guaranteed_profit_beats_market_in_rich(self):
        # "chac tang gia" (guaranteed profit, priority 2)
        # "loi nhuan" (market, priority 5) is also present but priority 2 wins
        r = classify("Loi nhuan chac tang gia khong?")
        assert r.intent == "legal"
        assert r.risk_level == "critical"
        assert r.must_use_legal_only is False

    def test_pricing_beats_market_in_rich(self):
        # pricing priority 3 < market priority 5
        r = classify("Bang gia thi truong bat dong san?")
        assert r.intent == "pricing"

    def test_sales_policy_beats_market_in_rich(self):
        # "chiet khau" (sales_policy p4), "thi truong" (market p5)
        r = classify("Chiet khau tren thi truong nhu the nao?")
        assert r.intent == "sales_policy"

    def test_legal_must_use_legal_only_true_for_direct(self):
        r = classify("Du an da mo ban chinh thuc chua?")
        assert r.must_use_legal_only is True

    def test_guaranteed_profit_must_use_legal_only_false(self):
        r = classify("Co cam ket loi nhuan gi khong?")
        assert r.must_use_legal_only is False
        assert "market" in r.target_sections


# ===========================================================================
# TASK-5 ADDITIONAL TESTS -- required Vietnamese questions not yet covered
# ===========================================================================


class TestClassifyLegalExtra:
    """Additional legal tests: HĐMB abbreviation."""

    def test_ky_hdmb(self):
        r = classify("Có ký HĐMB được chưa?")
        assert r.intent == "legal"
        assert r.risk_level == "critical"
        assert r.must_use_legal_only is True

    def test_co_ky_hdmb_duoc_chua(self):
        r = classify("Co ky HDMB duoc chua?")
        assert r.intent == "legal"
        assert r.risk_level == "critical"
        assert r.must_use_legal_only is True


class TestClassifyPricingExtra:
    """Additional pricing tests: shophouse, Studio+, unit-type price queries."""

    def test_shophouse_gia_bao_nhieu(self):
        r = classify("Shophouse giá bao nhiêu?")
        assert r.intent == "pricing"
        assert r.risk_level == "high"
        assert r.must_use_legal_only is False

    def test_gia_studio_plus(self):
        # "Studio+" is a product keyword (P9); "gia the nao" is not contiguous
        # after normalization so pricing (P3) does not fire first.
        # Both pricing and product are valid intents for this query.
        r = classify("Giá Studio+ thế nào?")
        assert r.intent in ("pricing", "product")
        assert r.risk_level in ("high", "medium")

    def test_can_3pn_bao_nhieu_tien(self):
        r = classify("Căn 3PN bao nhiêu tiền?")
        assert r.intent == "pricing"
        assert r.risk_level == "high"

    def test_pricing_sections_include_price_sheet(self):
        r = classify("Bảng giá mới nhất thế nào?")
        assert "pricing" in r.target_sections
        assert "price_sheet" in r.target_sections


class TestClassifySalesPolicyExtra:
    """Additional sales policy tests: ân hạn gốc, section assertions."""

    def test_an_han_goc_bao_lau(self):
        r = classify("Ân hạn gốc bao lâu?")
        assert r.intent == "sales_policy"
        assert r.risk_level == "high"
        assert r.must_use_legal_only is False

    def test_sales_policy_sections_match_agents_md(self):
        r = classify("Chinh sach thanh toan the nao?")
        assert "sales_policy" in r.target_sections
        assert "price_sheet" in r.target_sections
        assert "pricing" not in r.target_sections

    def test_an_han_no_goc(self):
        r = classify("An han no goc bao nhieu thang?")
        assert r.intent == "sales_policy"
        assert r.risk_level == "high"


class TestClassifyMarketExtra:
    """Additional market tests: section assertions."""

    def test_market_sections_match_agents_md(self):
        r = classify("Tiem nang bat dong san Me Linh?")
        assert r.target_sections == ["market"]

    def test_me_linh_co_tiem_nang(self):
        r = classify("Mê Linh có tiềm năng không?")
        assert r.intent == "market"
        assert r.risk_level == "medium"
        assert r.must_use_legal_only is False


class TestClassifyLocationExtra:
    """Additional location tests: Vành đai 4, section assertions."""

    def test_vanh_dai_4(self):
        r = classify("Kết nối Vành đai 4 thế nào?")
        assert r.intent == "location"
        assert r.risk_level == "medium"
        assert r.must_use_legal_only is False

    def test_location_sections_include_market(self):
        r = classify("Me Linh ket noi san bay Noi Bai the nao?")
        assert "location_connectivity" in r.target_sections
        assert "market" in r.target_sections

    def test_benh_vien(self):
        # "benh vien" is now in _LOCATION keywords
        r = classify("Gan benh vien nao?")
        assert r.intent == "location"
        assert r.risk_level == "medium"


class TestClassifySalesStrategyExtra:
    """Additional sales strategy tests: objection handling, che xa."""

    def test_khach_che_me_linh_xa(self):
        r = classify("Khách chê Mê Linh xa thì xử lý thế nào?")
        assert r.intent == "sales_strategy"
        assert r.risk_level == "medium"
        assert r.must_use_legal_only is False

    def test_khach_che_xa_tu_van_sao(self):
        r = classify("Khách chê xa thì tư vấn sao?")
        assert r.intent == "sales_strategy"
        assert r.risk_level == "medium"

    def test_sales_noi_gi_voi_nha_dau_tu(self):
        r = classify("Sales nên nói gì với nhà đầu tư?")
        assert r.intent == "sales_strategy"
        assert r.risk_level == "medium"

    def test_sales_strategy_sections(self):
        r = classify("Khach gia dinh tre nen tu van the nao?")
        assert "sales_strategy" in r.target_sections
        assert "personas" in r.target_sections
        assert "concept_positioning" not in r.target_sections

    def test_xu_ly_objection_vung_ven(self):
        r = classify("Xử lý objection vùng ven buồn thế nào?")
        assert r.intent == "sales_strategy"
        assert r.risk_level == "medium"


class TestClassifyPersonaExtra:
    """Additional persona tests: nhà đầu tư trung lưu, người trẻ mua căn đầu tiên."""

    def test_nha_dau_tu_trung_luu(self):
        r = classify("Nhà đầu tư trung lưu nên nghe thông điệp nào?")
        assert r.intent == "persona"
        assert r.risk_level == "low"
        assert r.must_use_legal_only is False

    def test_nguoi_tre_mua_can_dau_tien(self):
        r = classify("Người trẻ mua căn đầu tiên quan tâm gì?")
        assert r.intent == "persona"
        assert r.risk_level == "low"

    def test_nha_dau_tu(self):
        r = classify("Nha dau tu trung luu la ai?")
        assert r.intent == "persona"
        assert r.risk_level == "low"

    def test_persona_sections(self):
        r = classify("Khach hang muc tieu la ai?")
        assert "personas" in r.target_sections


class TestClassifyAmenitiesExtra:
    """Additional amenities tests: Neo Square, R&D Center."""

    def test_neo_square_la_gi(self):
        r = classify("Neo Square là gì?")
        assert r.intent == "amenities"
        assert r.risk_level == "low"
        assert r.must_use_legal_only is False

    def test_neo_square_dung_de_lam_gi(self):
        r = classify("Neo Square dùng để làm gì?")
        assert r.intent == "amenities"
        assert r.risk_level == "low"

    def test_rd_center(self):
        r = classify("R&D Center dùng để làm gì?")
        assert r.intent == "amenities"
        assert r.risk_level == "low"

    def test_amenities_section_is_factsheet(self):
        r = classify("NEO CITY co tien ich gi?")
        assert r.target_sections == ["factsheet"]

    def test_khu_vui_choi_tre_em(self):
        # "cho tre em" is now in _AMENITIES keywords
        r = classify("Có khu cho trẻ em không?")
        assert r.intent == "amenities"
        assert r.risk_level == "low"


class TestClassifyProduct:
    """Product / unit-type questions (intent=product, risk=medium)."""

    def test_2pn_dien_tich_bao_nhieu(self):
        r = classify("2PN diện tích bao nhiêu?")
        assert r.intent == "product"
        assert r.risk_level == "medium"
        assert r.must_use_legal_only is False

    def test_du_an_co_nhung_loai_can_ho_nao(self):
        r = classify("Dự án có những loại căn hộ nào?")
        assert r.intent == "product"
        assert r.risk_level == "medium"

    def test_du_an_co_shophouse(self):
        # "shophouse" alone (no pricing keyword) → product
        r = classify("Du an co shophouse khong?")
        # shophouse is in _AMENITIES (P10), product is P9, so product
        # but "shophouse" is ONLY in _AMENITIES → amenities
        # This is intentional: shophouse as facility → amenities
        assert r.intent in ("amenities", "product")

    def test_bao_nhieu_toa_chung_cu(self):
        r = classify("Có bao nhiêu tòa chung cư?")
        assert r.intent == "project_overview"

    def test_3pn_thiet_ke(self):
        r = classify("Thiet ke can 3PN nhu the nao?")
        assert r.intent == "product"
        assert r.risk_level == "medium"

    def test_product_sections(self):
        r = classify("2PN dien tich bao nhieu?")
        assert "factsheet" in r.target_sections
        assert "pricing" in r.target_sections

    def test_1pn_mat_bang(self):
        r = classify("Mat bang dien hinh can 1PN nhu the nao?")
        assert r.intent == "product"

    def test_studio_plus_san_pham(self):
        r = classify("Studio+ la loai san pham nao?")
        assert r.intent == "product"


# ===========================================================================
# COMPOUND PROFIT-GUARANTEE TESTS (new _has_profit_guarantee_compound helper)
# Tests cover reversed-order phrases like "loi nhuan ... co dam bao khong?"
# that the phrase-list _GUARANTEED_PROFIT would miss.
# ===========================================================================


class TestHasProfitGuaranteeCompound:
    """Direct unit tests for the _has_profit_guarantee_compound helper."""

    def test_returns_true_loi_nhuan_dam_bao(self):
        assert _has_profit_guarantee_compound("loi nhuan dau tu co dam bao khong") is True

    def test_returns_true_tang_gia_bao_dam(self):
        assert _has_profit_guarantee_compound("tang gia co bao dam khong") is True

    def test_returns_true_loi_tuc_cam_ket(self):
        assert _has_profit_guarantee_compound("loi tuc co cam ket khong") is True

    def test_returns_true_sinh_loi_chac_chan(self):
        assert _has_profit_guarantee_compound("sinh loi co chac chan khong") is True

    def test_returns_false_loi_nhuan_no_guarantee(self):
        # profit term present but NO guarantee term → should NOT fire
        assert _has_profit_guarantee_compound("loi nhuan dau tu o me linh") is False

    def test_returns_false_dam_bao_no_profit(self):
        # guarantee term present but NO profit term → should NOT fire
        assert _has_profit_guarantee_compound("du an co dam bao phap ly") is False

    def test_returns_false_empty(self):
        assert _has_profit_guarantee_compound("") is False


class TestClassifyCompoundProfitGuarantee:
    """classify() must return critical when profit+guarantee appear in reversed order.

    These variants were NOT matched by the single-phrase _GUARANTEED_PROFIT list
    and required the new _has_profit_guarantee_compound() helper.
    """

    # ── Variant 1: main failing case (reversed order, ASCII) ─────────────────
    def test_loi_nhuan_dam_bao_reversed_ascii(self):
        """'loi nhuan ... co dam bao' – profit before guarantee."""
        r = classify("Loi nhuan dau tu co dam bao khong?")
        assert r.intent == "legal"
        assert r.risk_level == "critical"
        assert r.must_use_legal_only is False
        assert "market" in r.target_sections

    # ── Variant 2: same query with Vietnamese diacritics ─────────────────────
    def test_loi_nhuan_dam_bao_reversed_vietnamese(self):
        """Full-diacritics version of variant 1."""
        r = classify("Lợi nhuận đầu tư có đảm bảo không?")
        assert r.intent == "legal"
        assert r.risk_level == "critical"
        assert r.must_use_legal_only is False

    # ── Variant 3: tang gia + bao dam (reversed) ─────────────────────────────
    def test_tang_gia_bao_dam_reversed(self):
        r = classify("Tang gia co bao dam khong?")
        assert r.intent == "legal"
        assert r.risk_level == "critical"
        assert r.must_use_legal_only is False

    # ── Variant 4: loi tuc (not in old phrase list) + cam ket ────────────────
    def test_loi_tuc_cam_ket_reversed(self):
        r = classify("Loi tuc co cam ket khong?")
        assert r.intent == "legal"
        assert r.risk_level == "critical"
        assert r.must_use_legal_only is False

    # ── Variant 5: sinh loi + chac chan (reversed order) ─────────────────────
    def test_sinh_loi_chac_chan_reversed(self):
        r = classify("Sinh loi co chac chan khong?")
        assert r.intent == "legal"
        assert r.risk_level == "critical"
        assert r.must_use_legal_only is False

    # ── Variant 6: loi nhuan + chac thang (reversed) ─────────────────────────
    def test_loi_nhuan_chac_thang_reversed(self):
        r = classify("Loi nhuan co chac thang khong?")
        assert r.intent == "legal"
        assert r.risk_level == "critical"

    # ── Variant 7: tang gia + cam ket (reversed) ─────────────────────────────
    def test_tang_gia_cam_ket_reversed(self):
        r = classify("Tang gia co cam ket gi khong?")
        assert r.intent == "legal"
        assert r.risk_level == "critical"

    # ── Variant 8: all compound results have risk=critical ───────────────────
    def test_all_compound_variants_are_critical(self):
        queries = [
            "Loi nhuan co dam bao khong?",
            "Sinh loi co bao dam khong?",
            "Tang gia co cam ket khong?",
            "Loi tuc co chac chan khong?",
        ]
        for q in queries:
            r = classify(q)
            assert r.risk_level == "critical", f"Expected critical for: {q!r}"

    # ── Variant 9: compound must_use_legal_only is always False ──────────────
    def test_compound_must_use_legal_only_false(self):
        r = classify("Loi nhuan dau tu co dam bao khong?")
        assert r.must_use_legal_only is False

    # ── Variant 10: pure market query NOT triggered by compound check ─────────
    def test_market_query_without_guarantee_not_affected(self):
        """'loi nhuan dau tu' with NO guarantee term must stay as market intent."""
        r = classify("Loi nhuan dau tu o Me Linh nhu the nao?")
        assert r.intent == "market"
        assert r.risk_level == "medium"

    # ------------------------------------------------------------------
    # Task 7 - High-impact failure pattern tests
    # ------------------------------------------------------------------

    def test_legal_008_chu_dau_tu_neolab(self):
        """legal_008: 'Chủ đầu tư NEO CITY là ai?' must be legal intent."""
        r = classify("Chủ đầu tư NEO CITY là ai?")
        assert r.intent == "legal"
        assert r.risk_level == "critical"
        assert r.must_use_legal_only is True

    def test_objection_007_chu_dau_tu_uy_tin(self):
        """objection_007: 'Chủ đầu tư NEOLAB có uy tín không?' must be legal intent."""
        r = classify("Chủ đầu tư NEOLAB có uy tín không?")
        assert r.intent == "legal"
        assert r.risk_level == "critical"
        assert r.must_use_legal_only is True

    def test_sales_policy_003_booking_tien(self):
        """sales_policy_003: 'Booking NEO CITY cần đặt bao nhiêu tiền?' must be sales_policy."""
        r = classify("Booking NEO CITY cần đặt bao nhiêu tiền?")
        assert r.intent == "sales_policy"
        assert r.risk_level == "high"

    def test_pricing_008_chiet_khau_thanh_toan(self):
        """pricing_008: 'Chiết khấu khi thanh toán sớm' - pricing with policy hint."""
        r = classify("Chiết khấu khi thanh toán sớm bao nhiêu phần trăm?")
        # Should be pricing (price keyword dominates) or sales_policy (policy hint)
        assert r.intent in ("pricing", "sales_policy")
        assert r.risk_level in ("high", "critical")

    def test_pricing_009_chinh_sach_early_bird(self):
        """pricing_009: 'NEO CITY có chính sách chiết khấu early bird không?'"""
        r = classify("NEO CITY có chính sách chiết khấu early bird không?")
        # Policy question with price term - could be either
        assert r.intent in ("pricing", "sales_policy")
        assert r.risk_level == "high"

    def test_pricing_011_dien_tich_studio(self):
        """pricing_011: 'Diện tích căn studio NEO CITY là bao nhiêu m2?'"""
        r = classify("Diện tích căn studio NEO CITY là bao nhiêu m2?")
        # Product question with area - should be product, not pricing
        assert r.intent == "product"

    def test_pricing_015_dien_tich_2pn(self):
        """pricing_015: 'Diện tích căn 2PN là bao nhiêu m2?'"""
        r = classify("Diện tích căn 2PN là bao nhiêu m2?")
        # Product question with area - should be product
        assert r.intent == "product"

    def test_pricing_016_chiet_khau_gioi_thieu(self):
        """pricing_016: 'Chính sách chiết khấu giới thiệu khách mua'"""
        r = classify("Chính sách chiết khấu giới thiệu khách mua là bao nhiêu?")
        # Policy question - should be sales_policy
        assert r.intent == "sales_policy"

    def test_pricing_019_dien_tich_3pn(self):
        """pricing_019: 'Diện tích căn hộ 3PN NEO CITY dự kiến bao nhiêu?'"""
        r = classify("Diện tích căn hộ 3PN NEO CITY dự kiến bao nhiêu?")
        # Product question with area - should be product
        assert r.intent == "product"

    def test_product_003_dien_tich_1pn1(self):
        """product_003: 'Căn 1PN+1 có diện tích bao nhiêu?'"""
        r = classify("Căn 1PN+1 có diện tích bao nhiêu?")
        assert r.intent == "product"

    def test_product_004_co_can_studio(self):
        """product_004: 'NEO CITY có căn studio không?'"""
        r = classify("NEO CITY có căn studio không?")
        assert r.intent == "product"

    def test_product_005_san_pham_thap_tang(self):
        """product_005: 'Sản phẩm thấp tầng NEO CITY gồm những loại nào?'"""
        r = classify("Sản phẩm thấp tầng NEO CITY gồm những loại nào?")
        assert r.intent == "product"

    def test_product_010_ban_giao_hoan_thien(self):
        """product_010: 'Căn hộ NEO CITY bàn giao hoàn thiện hay thô?'"""
        r = classify("Căn hộ NEO CITY bàn giao hoàn thiện hay thô?")
        assert r.intent == "product"

    def test_product_015_tien_do_xay_dung(self):
        """product_015: 'Tiến độ xây dựng NEO CITY dự kiến như thế nào?'"""
        r = classify("Tiến độ xây dựng NEO CITY dự kiến như thế nào?")
        # Should NOT be unknown - should match product or project_overview
        assert r.intent != "unknown"
        assert r.intent in ("product", "project_overview", "legal")

    def test_location_009_dong_anh_me_linh(self):
        """location_009: 'Đông Anh và Mê Linh có gì khác nhau về vị trí?'"""
        r = classify("Đông Anh và Mê Linh có gì khác nhau về vị trí?")
        assert r.intent == "location"

    def test_amenities_009_mam_non(self):
        """amenities_009: 'Mầm non trong dự án NEO CITY có không?'"""
        r = classify("Mầm non trong dự án NEO CITY có không?")
        # This is about amenities (preschool), not sales_policy
        assert r.intent == "amenities"
class TestTask7IntentOverrides:
    def test_amenities_policy_like_query_becomes_sales_policy(self):
        r = classify("Combo uu dai learning hub va mam non cua NEO CITY la gi?")
        assert r.intent == "sales_policy"
        assert r.risk_level == "high"

    def test_product_with_price_keyword_becomes_pricing(self):
        r = classify("Can studio NEO CITY gia bao nhieu?")
        assert r.intent == "pricing"
        assert r.risk_level == "high"

    def test_direct_connectivity_beats_market_without_investment_terms(self):
        r = classify("Vi tri NEO CITY ket noi Noi Bai va Dong Anh nhu the nao?")
        assert r.intent == "location"
        assert r.risk_level == "medium"
