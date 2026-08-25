from growth_data_agent.contracts import ProvisionalMetricInput
from growth_data_agent.policy import resolve_access_profile, tenant_ids_for_segment


def test_all_synthetic_agent_users_resolve_to_explicit_cross_source_profiles() -> None:
    expected = {
        "data_analyst": {
            "products": ("Jira", "Confluence"),
            "regions": ("Americas", "APAC", "EMEA"),
            "has_direct_identifier_entitlement": False,
        },
        "apac_regional_manager": {
            "products": ("Jira", "Confluence"),
            "regions": ("APAC",),
            "has_direct_identifier_entitlement": False,
        },
        "jira_product_manager": {
            "products": ("Jira",),
            "regions": ("Americas", "APAC", "EMEA"),
            "has_direct_identifier_entitlement": False,
        },
        "confluence_product_manager": {
            "products": ("Confluence",),
            "regions": ("Americas", "APAC", "EMEA"),
            "has_direct_identifier_entitlement": False,
        },
        "customer_success_manager": {
            "products": ("Jira", "Confluence"),
            "regions": ("APAC",),
            "has_direct_identifier_entitlement": True,
        },
    }

    for agent_user_id, profile_expectations in expected.items():
        profile = resolve_access_profile(agent_user_id)

        assert profile.products == profile_expectations["products"]
        assert profile.regions == profile_expectations["regions"]
        assert bool(profile.permitted_identifiers) is profile_expectations[
            "has_direct_identifier_entitlement"
        ]
        assert profile.permitted_columns
        assert profile.permitted_classifications
        assert profile.permitted_tenant_ids


def test_cross_source_filters_are_derived_from_the_same_profile_scope() -> None:
    profile = resolve_access_profile("apac_regional_manager")

    metric_constraints = profile.metricflow_where_constraints("Jira")
    document_filter = profile.evidence_filter("Jira", "APAC")
    graph_filter = profile.graph_filter("Jira", "APAC")

    assert "product_user__region IN ('APAC')" in metric_constraints
    assert document_filter.regions == graph_filter.regions == ("APAC",)
    assert document_filter.tenant_ids == graph_filter.tenant_ids
    assert (
        document_filter.identifier_entitlements
        == graph_filter.identifier_entitlements
        == ("none",)
    )


def test_direct_identifier_columns_are_explicitly_entitled() -> None:
    analyst = resolve_access_profile("data_analyst")
    customer_success = resolve_access_profile("customer_success_manager")
    tenant_id = ProvisionalMetricInput(name="tenant_id", source="Tenant dimension")

    assert analyst.permits_provisional_inputs([tenant_id]) is False
    assert customer_success.permits_provisional_inputs([tenant_id]) is True
    assert "tenant_id" not in analyst.permitted_columns
    assert "tenant_id" in customer_success.permitted_columns


def test_segment_graph_and_document_filters_share_the_exact_tenant_scope() -> None:
    profile = resolve_access_profile("data_analyst")

    document_filter = profile.evidence_filter("Confluence", "Americas", seat_tier="11-50")
    graph_filter = profile.graph_filter("Confluence", "Americas", seat_tier="11-50")

    assert document_filter.seat_tiers == graph_filter.seat_tiers == ("11-50",)
    assert document_filter.tenant_ids == graph_filter.tenant_ids == tenant_ids_for_segment(
        "Americas", "11-50"
    )


def test_customer_success_manager_filter_is_bounded_to_its_direct_identifier_portfolio() -> None:
    profile = resolve_access_profile("customer_success_manager")

    document_filter = profile.evidence_filter("Jira", "APAC")
    graph_filter = profile.graph_filter("Jira", "APAC")

    assert document_filter.tenant_ids == graph_filter.tenant_ids
    assert "tenant-0011" in document_filter.tenant_ids
    assert "tenant-0002" not in document_filter.tenant_ids
    assert document_filter.classifications == ("internal", "restricted")
    assert document_filter.identifier_entitlements == ("none", "direct")
