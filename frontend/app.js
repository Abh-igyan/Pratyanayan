const API_BASE = '/';

const orderForm = document.getElementById('order-form');
const checkoutButton = document.getElementById('checkout-button');
const orderStatus = document.getElementById('order-status');
const orderDetails = document.getElementById('order-details');
const recoveryList = document.getElementById('recovery-list');
const historyTableBody = document.getElementById('history-table-body');
const metricsContainer = document.getElementById('metrics');

let razorpayKeyId = '';
let razorpayCheckout = null;
let currentOrder = null;
let isVoiceEnabled = false;
let lastPlayedKey = null;

function updateVoiceUI() {
  const toggleBtn = document.getElementById('voice-toggle-btn');
  const iconSpan = document.getElementById('voice-icon');
  const labelSpan = document.getElementById('voice-label');
  const statusBadge = document.getElementById('voice-status-badge');
  if (!toggleBtn || !iconSpan || !labelSpan || !statusBadge) return;

  if (isVoiceEnabled) {
    iconSpan.innerText = '🔊';
    labelSpan.innerText = 'Voice On (Mute)';
    toggleBtn.classList.remove('secondary');
    toggleBtn.classList.add('primary');
    statusBadge.className = 'badge success';
    statusBadge.innerText = 'Voice Active';
  } else {
    iconSpan.innerText = '🔇';
    labelSpan.innerText = 'Enable Agent Voice';
    toggleBtn.classList.remove('primary');
    toggleBtn.classList.add('secondary');
    statusBadge.className = 'badge neutral';
    statusBadge.innerText = 'Voice Off';
  }
}

function stopVoicePlayback() {
  if ('speechSynthesis' in window) {
    window.speechSynthesis.cancel();
  }
}

async function playAgentVoice(orderId, messageKey = null) {
  if (!orderId) return;
  try {
    const response = await apiFetch(`payments/order/${orderId}/voice-recovery`);
    if (!response || !response.message) return;

    // Render latest spoken message in the active voice bar
    const liveBox = document.getElementById('voice-live-banner');
    if (liveBox) {
      const sourceTag = response.source === 'llm'
        ? `<span style="background:#d1fae5;color:#065f46;font-size:0.75rem;padding:2px 6px;border-radius:4px;margin-right:6px;">via Groq LLM</span>`
        : `<span style="background:#f1f5f9;color:#64748b;font-size:0.75rem;padding:2px 6px;border-radius:4px;margin-right:6px;">fallback</span>`;
      liveBox.innerHTML = `<strong>Agent Voice:</strong> ${sourceTag} <em>"${response.message}"</em>`;
      liveBox.style.display = 'block';
    }

    // Only speak aloud if voice is enabled by explicit user gesture
    if (isVoiceEnabled && 'speechSynthesis' in window) {
      window.speechSynthesis.cancel(); // Cancel any existing speech to prevent overlap
      // Strip any stray emojis defensively before passing to TTS
      const cleanText = response.message.replace(/[\u{1F300}-\u{1FAFF}|\u{2600}-\u{27BF}]/gu, '').trim();
      const utterance = new SpeechSynthesisUtterance(cleanText || response.message);
      utterance.lang = 'hi-IN';
      utterance.rate = 0.95;
      window.speechSynthesis.speak(utterance);
      if (messageKey) {
        lastPlayedKey = messageKey;
      }
    }
  } catch (err) {
    console.warn('Agent voice retrieval error:', err);
  }
}

function formatINR(paise) {
  return '₹' + (Number(paise) / 100).toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

async function apiFetch(endpoint, options = {}) {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers: { 'Content-Type': 'application/json', ...options.headers }
  });
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `API error: ${response.status}`);
  }
  return response.json();
}

async function loadRazorpayConfig() {
  if (razorpayKeyId) return;
  try {
    const config = await apiFetch('config');
    razorpayKeyId = config.razorpay_key_id || '';
  } catch (error) {
    throw new Error(`Could not load Razorpay config: ${error.message}`);
  }
}

async function createOrder(event) {
  event.preventDefault();
  const customerId = document.getElementById('customer-id').value;
  const amount = Number(document.getElementById('amount').value) * 100; // convert to paise

  try {
    await loadRazorpayConfig();
    const order = await apiFetch('orders', {
      method: 'POST',
      body: JSON.stringify({ customer_id: customerId, amount })
    });
    currentOrder = order;
    renderOrder(order);
    orderStatus.innerHTML = `Order created: ${order.razorpay_order_id}`;
    await refreshDashboard();
  } catch (error) {
    orderStatus.innerHTML = `<span class="bad">${error.message}</span>`;
  }
}

function renderOrder(order) {
  const isPaid = (order.status || '').toLowerCase() === 'paid';
  const statusBadge = isPaid
    ? '<span class="badge success">PAID</span>'
    : `<span class="badge neutral">${order.status}</span>`;
  const paidSuccessMsg = isPaid
    ? '<div style="margin-top: 8px; color: var(--success); font-weight: 600; display: flex; align-items: center; gap: 4px;">✓ Payment completed successfully</div>'
    : '';

  orderDetails.innerHTML = `
    <div style="font-size: 0.9em; background: ${isPaid ? '#f0fdf4' : '#f9fafb'}; padding: 12px; border-radius: 8px; border: 1px solid ${isPaid ? '#bbf7d0' : '#e5e7eb'};">
      <div style="margin-bottom: 6px;"><strong>Order ID:</strong> ${order.razorpay_order_id}</div>
      <div style="margin-bottom: 6px;"><strong>Amount:</strong> ${formatINR(order.amount)}</div>
      <div><strong>Status:</strong> ${statusBadge}</div>
      ${paidSuccessMsg}
    </div>
  `;

  if (isPaid) {
    checkoutButton.classList.add('hidden');
    checkoutButton.disabled = true;
  } else {
    checkoutButton.classList.remove('hidden');
    checkoutButton.disabled = false;
  }
}

async function loadActiveRecoveryState() {
  if (!currentOrder) {
    recoveryList.innerHTML = '<div style="color: var(--muted); text-align: center; padding: 40px;">No active failure selected. Create an order to see AI analysis.</div>';
    return;
  }

  try {
    const traces = await apiFetch(`payments/order/${currentOrder.razorpay_order_id}/trace`);
    if (!traces || traces.length === 0) {
      recoveryList.innerHTML = '<div style="color: var(--muted); text-align: center; padding: 40px;">No agent trace yet for this order. Waiting for events...</div>';
      return;
    }

    let html = '';
    
    // Process each trace event as a separate phase box
    traces.forEach((trace, idx) => {
      const isFailed = trace.event === 'payment.failed';
      const isCaptured = trace.event === 'payment.captured';
      
      let badgeColor = 'neutral';
      if (isFailed) badgeColor = 'danger';
      if (isCaptured) badgeColor = 'success';
      
      let evGrid = '';
      if (trace.probability !== null) {
        evGrid = `
          <div class="ev-grid" style="margin-top: 10px; margin-bottom: 10px;">
            <div class="ev-stat"><span class="ev-stat-label">Probability</span><span class="ev-stat-value">${(trace.probability * 100).toFixed(1)}%</span></div>
            <div class="ev-stat"><span class="ev-stat-label">Expected Val</span><span class="ev-stat-value">${formatINR(trace.expected_value)}</span></div>
            <div class="ev-stat"><span class="ev-stat-label">Net EV</span><span class="ev-stat-value">${formatINR(trace.net_ev)}</span></div>
          </div>
        `;
      }
      
      let planBlock = '';
      if (trace.selected_action) {
         let planColor = 'neutral';
         if (trace.selected_action === 'RETRY') planColor = 'success';
         if (trace.selected_action === 'WAIT') planColor = 'warning';
         if (trace.selected_action === 'STOP') planColor = 'danger';
         if (trace.selected_action === 'SWITCH_PAYMENT_METHOD') planColor = 'primary';
         const planLabel = trace.selected_action === 'SWITCH_PAYMENT_METHOD' ? 'Recommended method' : 'Plan';
         planBlock = `<div style="margin-top: 8px;"><strong>${planLabel}:</strong> <span class="badge ${planColor}">${trace.selected_action === 'SWITCH_PAYMENT_METHOD' ? 'TRY_METHOD' : trace.selected_action}</span>${trace.planned_method ? ` → <strong>${trace.planned_method.toUpperCase()}</strong>` : ''}</div>`;
      }

      let verifyBlock = '';
      if (trace.webhook_outcome) {
         verifyBlock = `<div style="margin-top: 8px;"><strong>Verified:</strong> <span class="badge ${trace.webhook_outcome === 'RECOVERY_SUCCESS' ? 'success' : 'neutral'}">${trace.webhook_outcome}</span></div>`;
      }
      if (trace.actual_payment_method) {
        verifyBlock += `<div style="margin-top: 8px;"><strong>Actual payment method:</strong> ${trace.actual_payment_method.toUpperCase()}</div>`;
      }
      if (trace.decision_inputs) {
        const attemptNum = trace.decision_inputs?.base_case?.attempt_number;
        if (attemptNum) {
          verifyBlock += `<div style="margin-top: 8px; color: var(--muted); font-size: 0.85rem;"><strong>Current order:</strong> Attempt ${attemptNum}</div>`;
        }
      }
      if (trace.customer_history && trace.planned_method) {
        const totalPrior = trace.customer_history.total_attempts || 0;
        const methodRate = trace.customer_history.method_success_rates?.[trace.planned_method] ?? 0;
        if (totalPrior > 0) {
          verifyBlock += `<div style="margin-top: 6px; color: var(--muted); font-size: 0.85rem;">Customer history: ${totalPrior} previous-order attempt(s) &mdash; ${trace.planned_method.toUpperCase()} prior success rate: ${(methodRate * 100).toFixed(0)}%</div>`;
        } else {
          verifyBlock += `<div style="margin-top: 6px; color: var(--muted); font-size: 0.85rem;">Customer history: no previous-order attempts on record</div>`;
        }
      }
      if (trace.recovered_revenue) {
         verifyBlock += `<div style="margin-top: 8px; font-weight: bold; color: var(--success);">✅ Recovered: ${formatINR(trace.recovered_revenue)}</div>`;
      }
      
      let retryBtn = '';
      const orderIsPaid = currentOrder && (currentOrder.status || '').toLowerCase() === 'paid';
      // We only show retry button if order is NOT paid, this is a failed/abandoned event, the action is RETRY or SWITCH_PAYMENT_METHOD, and this is the last trace in the list.
      if (!orderIsPaid && (isFailed || trace.event === 'checkout.abandoned') && (trace.selected_action === 'RETRY' || trace.selected_action === 'SWITCH_PAYMENT_METHOD') && idx === traces.length - 1) {
         retryBtn = `<button data-retry="${currentOrder.razorpay_order_id}" class="retry-button" style="width: 100%; margin-top: 16px; padding: 12px; background-color: ${trace.selected_action === 'SWITCH_PAYMENT_METHOD' ? 'var(--primary)' : 'var(--success)'}; color: white; border: none; border-radius: 4px; cursor: pointer;">Execute Plan (${trace.selected_action === 'SWITCH_PAYMENT_METHOD' ? `Switch to ${trace.planned_method}` : 'Retry Payment'})</button>`;
      }

      html += `
        <div class="ai-card" style="border-left: 4px solid var(--${badgeColor});">
          <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
            <div style="font-weight: 600; font-size: 1.05rem;">Agent Step ${idx + 1}: ${trace.event}</div>
            <div style="font-size: 0.8rem; color: var(--muted);">${new Date(trace.created_at).toLocaleTimeString()}</div>
          </div>
          
          <div style="font-size: 0.9rem; color: #475569; margin-bottom: 8px;">
            <strong>Diagnosed:</strong> ${trace.diagnosis || 'N/A'}
          </div>
          
          ${evGrid}
          ${planBlock}
          ${verifyBlock}
          
          ${trace.execution_result ? `<div style="font-size: 0.85rem; color: var(--muted); margin-top: 8px;"><em>Execution: ${trace.execution_result}</em></div>` : ''}
          ${retryBtn}
        </div>
      `;
    });

    recoveryList.innerHTML = html;

    // Attach retry button listeners
    const retryButtons = recoveryList.querySelectorAll('.retry-button');
    retryButtons.forEach((button) => {
      button.addEventListener('click', async () => {
        try {
          button.disabled = true;
          button.innerText = 'Processing...';
          const result = await apiFetch(`payments/order/${button.dataset.retry}/retry`, { method: 'POST' });
          orderStatus.innerHTML = `<span class="warn">${result.message}</span>`;
          openCheckout(currentOrder.razorpay_order_id);
        } catch (error) {
          orderStatus.innerHTML = `<span class="bad">${error.message}</span>`;
          button.disabled = false;
          button.innerText = 'Execute Plan (Retry Payment)';
        }
      });
    });

    // Check if the latest trace represents a new agent recovery decision that should be voiced
    const latestTrace = traces[traces.length - 1];
    if (latestTrace) {
      const messageKey = `${currentOrder.razorpay_order_id}_${latestTrace.id || latestTrace.event}_${latestTrace.selected_action || ''}`;
      if (lastPlayedKey !== messageKey) {
        if (isVoiceEnabled) {
          await playAgentVoice(currentOrder.razorpay_order_id, messageKey);
        } else {
          // If voice is off, still update the banner visually so the merchant can see the text guidance
          try {
            const response = await apiFetch(`payments/order/${currentOrder.razorpay_order_id}/voice-recovery`);
            const liveBox = document.getElementById('voice-live-banner');
            if (liveBox && response && response.message) {
              const sourceTag = response.source === 'llm'
                ? `<span style="background:#d1fae5;color:#065f46;font-size:0.75rem;padding:2px 6px;border-radius:4px;margin-right:6px;">via Groq LLM</span>`
                : `<span style="background:#f1f5f9;color:#64748b;font-size:0.75rem;padding:2px 6px;border-radius:4px;margin-right:6px;">fallback</span>`;
              liveBox.innerHTML = `<strong>Agent Voice:</strong> ${sourceTag} <em>"${response.message}"</em>`;
              liveBox.style.display = 'block';
            }
          } catch (e) {
            // non-critical
          }
        }
      }
    }
  } catch (error) {
    recoveryList.innerHTML = `<span class="bad">Error loading trace: ${error.message}</span>`;
  }
}

async function loadCandidatesTable() {
  try {
    const candidates = await apiFetch('payments/candidates');
    if (!candidates || candidates.length === 0) {
      historyTableBody.innerHTML = '<tr><td colspan="9" style="text-align: center; color: var(--muted); padding: 24px;">No recovery candidates found.</td></tr>';
      return;
    }

    historyTableBody.innerHTML = candidates.map(c => {
      const prob = c.recovery_probability ? (Number(c.recovery_probability) * 100).toFixed(1) + '%' : '-';
      const ev = c.expected_recovered_value ? formatINR(c.expected_recovered_value) : '-';
      const netEv = c.net_expected_value ? formatINR(c.net_expected_value) : '-';
      
      let decisionBadge = `<span class="badge neutral">${c.ai_decision || 'N/A'}</span>`;
      if (c.ai_decision === 'RETRY') decisionBadge = `<span class="badge success">RETRY</span>`;
      if (c.ai_decision === 'WAIT') decisionBadge = `<span class="badge warning">WAIT</span>`;
      if (c.ai_decision === 'STOP') decisionBadge = `<span class="badge danger">STOP</span>`;
      if (c.ai_decision === 'SWITCH_PAYMENT_METHOD') decisionBadge = `<span class="badge primary">SWITCH</span>`;

      let statusBadge = `<span class="badge neutral">${c.status}</span>`;
      if (c.status === 'Recovered') statusBadge = `<span class="badge success">Recovered</span>`;
      if (c.status === 'Exhausted' || c.status === 'Stopped') statusBadge = `<span class="badge danger">${c.status}</span>`;
      if (c.status === 'Paid') statusBadge = `<span class="badge success">Paid</span>`;
      if (c.status === 'Recoverable') statusBadge = `<span class="badge warning">Recoverable</span>`;
      
      const isPaid = c.status === 'Paid' || c.status === 'Recovered';
      const isRetryable = !isPaid && (c.ai_decision === 'RETRY' || c.ai_decision === 'SWITCH_PAYMENT_METHOD') && c.status === 'Recoverable';
      const actionBtn = isRetryable ? `<button class="secondary action-retry" data-id="${c.razorpay_order_id}" style="padding: 6px 12px; font-size: 0.8rem;">Action</button>` : '-';

      return `
        <tr>
          <td style="font-family: monospace; font-size: 0.85em;">${c.razorpay_order_id}</td>
          <td>${formatINR(c.amount)}</td>
          <td style="font-size: 0.85em; color: var(--danger);">${c.failure_code}</td>
          <td>${prob}</td>
          <td>${ev}</td>
          <td style="font-weight: 600;">${netEv}</td>
          <td>${decisionBadge}</td>
          <td>${statusBadge}</td>
          <td>${actionBtn}</td>
        </tr>
      `;
    }).join('');

    // Attach listeners to table retry buttons
    document.querySelectorAll('.action-retry').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        const orderId = e.target.dataset.id;
        try {
          e.target.disabled = true;
          e.target.innerText = '...';
          const result = await apiFetch(`payments/order/${orderId}/retry`, { method: 'POST' });
          alert(result.message);
          openCheckout(orderId);
        } catch (error) {
          alert(`Error: ${error.message}`);
          e.target.disabled = false;
          e.target.innerText = 'Retry';
        }
      });
    });

  } catch (error) {
    historyTableBody.innerHTML = `<tr><td colspan="9" class="bad">Error loading candidates: ${error.message}</td></tr>`;
  }
}

async function loadMetrics() {
  try {
    const metrics = await apiFetch('dashboard/metrics');
    
    document.getElementById('metric-at-risk').innerText = formatINR(metrics.revenue_at_risk || 0);
    document.getElementById('metric-recovered').innerText = formatINR(metrics.recovered_revenue || 0);
    document.getElementById('metric-rate').innerText = ((metrics.revenue_recovery_rate || 0) * 100).toFixed(1) + '%';
    document.getElementById('metric-attempts').innerText = metrics.recovery_attempts || 0;
    document.getElementById('metric-active').innerText = metrics.active_recovery_candidates || 0;

    // Detailed Grid
    metricsContainer.innerHTML = `
      <div class="metric-card"><div class="metric-label">Total Orders</div><div class="metric-value">${metrics.total_orders}</div></div>
      <div class="metric-card"><div class="metric-label">Expected Revenue</div><div class="metric-value">${formatINR(metrics.total_expected_revenue)}</div></div>
      <div class="metric-card"><div class="metric-label">Successful Orders</div><div class="metric-value">${metrics.successful_orders}</div></div>
      <div class="metric-card"><div class="metric-label">Failed Attempts</div><div class="metric-value">${metrics.failed_payment_attempts}</div></div>
      <div class="metric-card"><div class="metric-label">Total Candidates</div><div class="metric-value">${metrics.recovery_cases_history ?? 0}</div></div>
      <div class="metric-card"><div class="metric-label">Exhausted Cases</div><div class="metric-value">${metrics.exhausted_escalated_cases}</div></div>
    `;
  } catch (error) {
    metricsContainer.innerHTML = `<span class="bad">${error.message}</span>`;
  }
}

async function refreshDashboard() {
  if (currentOrder && currentOrder.razorpay_order_id) {
    try {
      const statusRes = await apiFetch(`payments/order/${currentOrder.razorpay_order_id}/status`);
      if (statusRes && statusRes.paid) {
        currentOrder.status = 'paid';
        renderOrder(currentOrder);
      }
    } catch (e) {
      // non-critical
    }
  }
  await Promise.all([
    loadMetrics(),
    loadCandidatesTable(),
    loadActiveRecoveryState()
  ]);
}

async function openCheckout(orderId) {
  if (!orderId) {
    alert('Create an order first.');
    return;
  }

  if (currentOrder && (currentOrder.status || '').toLowerCase() === 'paid') {
    alert('This order is already paid.');
    return;
  }

  try {
    await loadRazorpayConfig();
    if (!razorpayKeyId) {
      throw new Error('Razorpay key is not configured.');
    }
  } catch (error) {
    orderStatus.innerHTML = `<span class="bad">${error.message}</span>`;
    return;
  }

  razorpayCheckout = new Razorpay({
    key: razorpayKeyId,
    amount: currentOrder.amount, // Note: if they retry a different order from table, this might be wrong, but in this demo they only retry currentOrder usually
    currency: currentOrder.currency,
    name: 'Revenue Recovery Demo',
    description: `Order ${currentOrder.internal_id}`,
    order_id: orderId,
    handler: async function (response) {
      orderStatus.innerHTML = `Payment success: ${response.razorpay_payment_id}`;
      const status = await apiFetch(`payments/order/${orderId}/status`);
      const paid = status.paid ? 'paid' : 'not yet confirmed';
      if (currentOrder && currentOrder.razorpay_order_id === orderId) {
        currentOrder.status = paid;
        renderOrder(currentOrder);
      }
      // Refresh dashboard after a short delay to allow webhooks to land
      setTimeout(refreshDashboard, 1000);
    },
    modal: {
      ondismiss: async function () {
        try {
          await apiFetch(`payments/order/${orderId}/abandon`, { method: 'POST' });
          orderStatus.innerHTML = 'Checkout abandoned. Recovery recommendation is ready.';
        } catch (error) {
          orderStatus.innerHTML = 'Checkout closed. Checking state in 2s...';
        }
        // Auto refresh after a 2s delay to catch async webhooks
        setTimeout(refreshDashboard, 2000);
      }
    },
    prefill: { name: 'Customer Demo', email: 'demo@example.com', contact: '9999999999' },
    theme: { color: '#3b82f6' }
  });

  razorpayCheckout.open();
}

checkoutButton.addEventListener('click', () => {
  if (currentOrder) openCheckout(currentOrder.razorpay_order_id);
});
orderForm.addEventListener('submit', createOrder);
document.getElementById('refresh-metrics').addEventListener('click', () => {
  document.getElementById('refresh-metrics').innerText = 'Refreshing...';
  refreshDashboard().finally(() => {
    document.getElementById('refresh-metrics').innerText = '⟳ Refresh Dashboard';
  });
});

// Global Voice toggle button listener (explicit user gesture)
const voiceToggleBtn = document.getElementById('voice-toggle-btn');
if (voiceToggleBtn) {
  voiceToggleBtn.addEventListener('click', () => {
    isVoiceEnabled = !isVoiceEnabled;
    updateVoiceUI();
    if (!isVoiceEnabled) {
      stopVoicePlayback();
    } else if (currentOrder) {
      // User explicitly enabled voice; trigger latest recovery voice message if available
      playAgentVoice(currentOrder.razorpay_order_id, 'manual_activation');
    }
  });
}

// Initial load
updateVoiceUI();
refreshDashboard();
