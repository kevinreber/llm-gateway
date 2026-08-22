package observe

import (
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"
)

// Every test here uses a provider label unique to itself. The metrics
// are process-wide singletons by design, so tests that shared a label
// would be asserting on each other's writes in whatever order the test
// binary happened to run them.

func TestSetBreakerState_PublishesBothGauges(t *testing.T) {
	// breaker_state and provider_health are the same fact at two
	// altitudes, so they must never disagree: a dashboard showing a
	// closed circuit next to an alert firing on unhealthy is worse than
	// either signal alone.
	cases := []struct {
		name       string
		provider   string
		state      float64
		wantHealth float64
	}{
		{"closed is healthy", "gauges-closed", BreakerClosed, 1},
		{"half-open is healthy", "gauges-halfopen", BreakerHalfOpen, 1},
		{"open is unhealthy", "gauges-open", BreakerOpen, 0},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			SetBreakerState(tc.provider, tc.state)

			if got := testutil.ToFloat64(breakerState.WithLabelValues(tc.provider)); got != tc.state {
				t.Errorf("breaker_state = %v, want %v", got, tc.state)
			}
			if got := testutil.ToFloat64(providerHealth.WithLabelValues(tc.provider)); got != tc.wantHealth {
				t.Errorf("provider_health = %v, want %v", got, tc.wantHealth)
			}
		})
	}
}

func TestSetBreakerState_RecoveryFlipsHealthBack(t *testing.T) {
	// A gauge that latched on the way down would keep a recovered
	// provider looking dead, and the alert built on it would have to be
	// silenced to be survivable — which is how it stops being an alert.
	const p = "gauges-recovery"

	SetBreakerState(p, BreakerOpen)
	if got := testutil.ToFloat64(providerHealth.WithLabelValues(p)); got != 0 {
		t.Fatalf("provider_health after open = %v, want 0", got)
	}

	SetBreakerState(p, BreakerClosed)
	if got := testutil.ToFloat64(providerHealth.WithLabelValues(p)); got != 1 {
		t.Errorf("provider_health after close = %v, want 1", got)
	}
}

func TestRecordRequest_CountsPerLabelSet(t *testing.T) {
	const p = "counter-provider"

	RecordRequest("alpha", p, ResultOK)
	RecordRequest("alpha", p, ResultOK)
	RecordRequest("alpha", p, ResultProviderError)

	if got := testutil.ToFloat64(requestsTotal.WithLabelValues("alpha", p, ResultOK)); got != 2 {
		t.Errorf("ok count = %v, want 2", got)
	}
	if got := testutil.ToFloat64(requestsTotal.WithLabelValues("alpha", p, ResultProviderError)); got != 1 {
		t.Errorf("provider_error count = %v, want 1", got)
	}
}

func TestAddCostCents_AccumulatesFractions(t *testing.T) {
	// Cents are fractional — a short completion costs a small fraction
	// of one — so a counter that rounded or truncated would report zero
	// spend for exactly the traffic pattern the gateway is built for.
	const p = "cost-provider"

	AddCostCents(p, "model-x", 0.25)
	AddCostCents(p, "model-x", 0.5)

	if got := testutil.ToFloat64(costCentsTotal.WithLabelValues(p, "model-x")); got != 0.75 {
		t.Errorf("cost_cents_total = %v, want 0.75", got)
	}
}

func TestAddCostCents_ZeroStillCreatesTheSeries(t *testing.T) {
	// An unpriced model records zero rather than nothing, so it shows up
	// on a cost dashboard as serving traffic it cannot account for
	// instead of being indistinguishable from a model nobody called.
	const p = "cost-unpriced"

	AddCostCents(p, "mystery-model", 0)

	if got := testutil.ToFloat64(costCentsTotal.WithLabelValues(p, "mystery-model")); got != 0 {
		t.Errorf("cost_cents_total = %v, want 0", got)
	}
	if n := testutil.CollectAndCount(costCentsTotal); n == 0 {
		t.Error("expected the zero-valued series to be collected")
	}
}

func TestObserveRequestDuration_LandsInABucket(t *testing.T) {
	const p = "hist-provider"

	ObserveRequestDuration("alpha", p, 750*time.Millisecond)

	if n := testutil.CollectAndCount(requestDuration); n == 0 {
		t.Fatal("no histogram series collected")
	}
}
