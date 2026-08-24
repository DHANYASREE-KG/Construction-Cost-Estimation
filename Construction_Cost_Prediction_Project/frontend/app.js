// Presets database
const PRESETS = {
    standard_3bhk: {
        Construction_Year: 2027,
        City: 'Chennai',
        Plot_Area_sqft: 2400,
        Builtup_Area_sqft: 2000,
        Number_of_Floors: 2,
        House_Type: 'Residential',
        Construction_Quality: 'Standard',
        Bedroom_Count: 3,
        Bathroom_Count: 3,
        Hall_Count: 1,
        Kitchen_Count: 1,
        Foundation_Type: 'RCC',
        Steel_Quantity_kg: 9000,
        Cement_Bags: 660,
        Sand_Quantity_m3: 30.0,
        Aggregate_Quantity_m3: 24.0,
        Brick_Quantity: 16000,
        Electrical_Points: 35,
        Plumbing_Points: 22,
        Mason_Labour_Days: 80,
        Carpenter_Labour_Days: 50,
        Electrician_Labour_Days: 25,
        Plumber_Labour_Days: 22,
        Cement_Rate_per_Bag: 420,
        Steel_Rate_per_kg: 82,
        Sand_Rate_per_m3: 2200,
        Brick_Rate_per_1000: 11000
    },
    luxury_villa: {
        Construction_Year: 2027,
        City: 'Coimbatore',
        Plot_Area_sqft: 4000,
        Builtup_Area_sqft: 3500,
        Number_of_Floors: 2,
        House_Type: 'Residential',
        Construction_Quality: 'Luxury',
        Bedroom_Count: 4,
        Bathroom_Count: 4,
        Hall_Count: 2,
        Kitchen_Count: 2,
        Foundation_Type: 'RCC',
        Steel_Quantity_kg: 16000,
        Cement_Bags: 1150,
        Sand_Quantity_m3: 52.0,
        Aggregate_Quantity_m3: 42.0,
        Brick_Quantity: 28000,
        Electrical_Points: 65,
        Plumbing_Points: 40,
        Mason_Labour_Days: 140,
        Carpenter_Labour_Days: 95,
        Electrician_Labour_Days: 45,
        Plumber_Labour_Days: 38,
        Cement_Rate_per_Bag: 440,
        Steel_Rate_per_kg: 85,
        Sand_Rate_per_m3: 2300,
        Brick_Rate_per_1000: 11500
    },
    economy_2bhk: {
        Construction_Year: 2027,
        City: 'Madurai',
        Plot_Area_sqft: 1400,
        Builtup_Area_sqft: 1100,
        Number_of_Floors: 1,
        House_Type: 'Residential',
        Construction_Quality: 'Economy',
        Bedroom_Count: 2,
        Bathroom_Count: 2,
        Hall_Count: 1,
        Kitchen_Count: 1,
        Foundation_Type: 'RCC',
        Steel_Quantity_kg: 4950,
        Cement_Bags: 360,
        Sand_Quantity_m3: 16.5,
        Aggregate_Quantity_m3: 13.2,
        Brick_Quantity: 8800,
        Electrical_Points: 20,
        Plumbing_Points: 12,
        Mason_Labour_Days: 45,
        Carpenter_Labour_Days: 28,
        Electrician_Labour_Days: 14,
        Plumber_Labour_Days: 12,
        Cement_Rate_per_Bag: 410,
        Steel_Rate_per_kg: 80,
        Sand_Rate_per_m3: 2100,
        Brick_Rate_per_1000: 10500
    }
};

document.addEventListener('DOMContentLoaded', () => {
    handlePredict(); // Run initial calculation
});

function switchTab(tabId) {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

    const activeBtn = Array.from(document.querySelectorAll('.tab-btn')).find(b => b.getAttribute('onclick').includes(tabId));
    if (activeBtn) activeBtn.classList.add('active');

    const targetContent = document.getElementById(tabId);
    if (targetContent) targetContent.classList.add('active');
}

function loadPreset(presetKey) {
    const data = PRESETS[presetKey];
    if (!data) return;

    document.querySelectorAll('.btn-preset').forEach(b => b.classList.remove('active'));
    const btn = Array.from(document.querySelectorAll('.btn-preset')).find(b => b.getAttribute('onclick').includes(presetKey));
    if (btn) btn.classList.add('active');

    for (const [k, v] of Object.entries(data)) {
        const input = document.getElementById(k);
        if (input) input.value = v;
    }

    handlePredict();
}

function autoCalculateQuantities() {
    const area = parseFloat(document.getElementById('Builtup_Area_sqft').value) || 2000;
    
    // Physical engineering estimation rules
    document.getElementById('Steel_Quantity_kg').value = Math.round(area * 4.5);
    document.getElementById('Cement_Bags').value = Math.round(area * 0.33);
    document.getElementById('Sand_Quantity_m3').value = (area * 0.015).toFixed(1);
    document.getElementById('Aggregate_Quantity_m3').value = (area * 0.012).toFixed(1);
    document.getElementById('Brick_Quantity').value = Math.round(area * 8);
    
    // Labour days
    document.getElementById('Mason_Labour_Days').value = Math.round(area * 0.04);
    document.getElementById('Carpenter_Labour_Days').value = Math.round(area * 0.025);
    document.getElementById('Electrician_Labour_Days').value = Math.round(area * 0.012);
    document.getElementById('Plumber_Labour_Days').value = Math.round(area * 0.011);
}

function getFormData() {
    return {
        Construction_Year: parseInt(document.getElementById('Construction_Year').value) || 2027,
        City: document.getElementById('City').value,
        Plot_Area_sqft: parseFloat(document.getElementById('Plot_Area_sqft').value) || 2400,
        Builtup_Area_sqft: parseFloat(document.getElementById('Builtup_Area_sqft').value) || 2000,
        Number_of_Floors: parseInt(document.getElementById('Number_of_Floors').value) || 2,
        House_Type: 'Residential',
        Construction_Quality: document.getElementById('Construction_Quality').value,
        Bedroom_Count: parseInt(document.getElementById('Bedroom_Count').value) || 3,
        Bathroom_Count: parseInt(document.getElementById('Bathroom_Count').value) || 3,
        Hall_Count: 1,
        Kitchen_Count: 1,
        Foundation_Type: 'RCC',
        Steel_Quantity_kg: parseFloat(document.getElementById('Steel_Quantity_kg').value) || 9000,
        Cement_Bags: parseFloat(document.getElementById('Cement_Bags').value) || 660,
        Sand_Quantity_m3: parseFloat(document.getElementById('Sand_Quantity_m3').value) || 30.0,
        Aggregate_Quantity_m3: parseFloat(document.getElementById('Aggregate_Quantity_m3').value) || 24.0,
        Brick_Quantity: parseFloat(document.getElementById('Brick_Quantity').value) || 16000,
        Electrical_Points: parseInt(document.getElementById('Electrical_Points').value) || 35,
        Plumbing_Points: parseInt(document.getElementById('Plumbing_Points').value) || 22,
        Mason_Labour_Days: parseFloat(document.getElementById('Mason_Labour_Days').value) || 80,
        Carpenter_Labour_Days: parseFloat(document.getElementById('Carpenter_Labour_Days').value) || 50,
        Electrician_Labour_Days: parseFloat(document.getElementById('Electrician_Labour_Days').value) || 25,
        Plumber_Labour_Days: parseFloat(document.getElementById('Plumber_Labour_Days').value) || 22,
        Cement_Rate_per_Bag: parseFloat(document.getElementById('Cement_Rate_per_Bag').value) || 420,
        Steel_Rate_per_kg: parseFloat(document.getElementById('Steel_Rate_per_kg').value) || 82,
        Sand_Rate_per_m3: parseFloat(document.getElementById('Sand_Rate_per_m3').value) || 2200,
        Brick_Rate_per_1000: parseFloat(document.getElementById('Brick_Rate_per_1000').value) || 11000
    };
}

async function handlePredict() {
    const payload = getFormData();
    const btn = document.getElementById('btnPredict');
    
    // Update live summary block
    document.getElementById('sumCity').innerText = payload.City;
    document.getElementById('sumArea').innerText = `${payload.Builtup_Area_sqft.toLocaleString()} sq.ft (${payload.Number_of_Floors} Floors)`;
    document.getElementById('sumQuality').innerText = payload.Construction_Quality;
    document.getElementById('sumRooms').innerText = `${payload.Bedroom_Count} BHK / ${payload.Bathroom_Count} Bath`;
    document.getElementById('targetYearBadge').innerText = `Year ${payload.Construction_Year}`;

    try {
        if (btn) {
            btn.innerHTML = '<i data-lucide="loader-2" class="spin"></i> Calculating...';
            if (window.lucide) lucide.createIcons();
        }

        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!res.ok) throw new Error('Prediction API error');
        const data = await res.json();

        // Update Hero DOM elements
        document.getElementById('costMain').innerText = data.best_estimate_inr;
        document.getElementById('costLakhs').innerText = `~${data.best_estimate_lakhs}`;
        document.getElementById('costPerSqft').innerText = data.cost_per_sqft;

    } catch (err) {
        console.error('Error predicting cost:', err);
    } finally {
        if (btn) {
            btn.innerHTML = '<i data-lucide="calculator"></i> Calculate Estimated Cost';
            if (window.lucide) lucide.createIcons();
        }
    }
}
