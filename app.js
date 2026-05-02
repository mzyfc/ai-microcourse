//============================================================
// 配置
// ============================================================
const BACKEND_URL = 'http://localhost:3000';
let currentQuizId = null;
let currentUserId = 1;
let currentCourseId = 1;
let chatCount = 0;

// 行为采集变量
let pauseCount = 0;
let rewatchCount = 0;
let seekCount = 0;

// ============================================================
// 工具函数
// ============================================================
function clamp(v, min = 0, max = 100) {
    return Math.max(min, Math.min(max, v));
}

function normalizeCount(value, maxValue) {
    return clamp((value / maxValue) * 100);
}

// ============================================================
// K-means 聚类引擎（真实迭代版 · 三维特征空间）
// ============================================================

// 初始质心 [学习投入度, 理解压力, 学习连续性]
const INITIAL_CENTERS = [
    [25, 20, 75],   // 新晋体验者
    [55, 80, 60],   // 基础薄弱型
    [75, 45, 80],   // 稳步提升型
    [90, 35, 90],   // 强化钻研型
    [45, 40, 30]    // 碎片跳学型
];

// 画像元数据
const PROFILE_META = [
    { id: 0, name: "新晋体验者", color: "#95a5a6", tags: ["初出茅庐","还在摸索"], score: 60,
      suggestion: "建议继续观看视频，了解机器学习的基础概念，不要跳过任何章节。",
      radar: [40, 30, 20, 50, 35] },
    { id: 1, name: "基础薄弱型", color: "#e74c3c", tags: ["遇到瓶颈","勤学好问"], score: 72,
      suggestion: "遇到难点不要怕，AI助教已记录你的高频问题，建议复习前置数学知识。",
      radar: [60, 75, 40, 55, 50] },
    { id: 2, name: "稳步提升型", color: "#f39c12", tags: ["渐入佳境","自主思考"], score: 85,
      suggestion: "理解得不错！尝试在脑海中构建算法知识框架，并尝试生成随堂测验。",
      radar: [80, 60, 70, 75, 65] },
    { id: 3, name: "强化钻研型", color: "#2ecc71", tags: ["融会贯通","学霸潜质"], score: 98,
      suggestion: "基础已熟练掌握，建议直接生成 AI 随堂测验挑战进阶题目！",
      radar: [95, 70, 90, 88, 92] },
    { id: 4, name: "碎片跳学型", color: "#9b59b6", tags: ["跳转频繁","路径不连续"], score: 68,
      suggestion: "检测到你的学习路径不够连续，建议减少快进和跳转，按课程顺序重新梳理本节核心概念。",
      radar: [55, 45, 40, 60, 50] }
];

const K = 5;
const FEATURE_WEIGHTS = [1.0, 1.1, 1.2];
const MAX_ITER = 20;
const CONVERGE_THRESHOLD = 0.01;
const RADAR_LABELS = ["理解力","专注度","互动性","进度","综合"];

// 历史数据点池
let dataPoints = [];

// 当前质心（会被迭代更新）
let currentCentroids = INITIAL_CENTERS.map(c => [...c]);

// 迭代日志
let iterationLog = [];

// ============================================================
// 三维特征构造
// ============================================================
function buildLearningFeatures() {
    const watchTime = parseFloat(document.getElementById('inputTime').value) || 0;
    const askCount = parseInt(document.getElementById('inputSpeak').value) || 0;

    const progressText = document.getElementById('progressText').textContent.replace('%', '');
    const progress = parseFloat(progressText) || 0;

    const video = document.getElementById('myVideo');
    const duration = (video && video.duration && !isNaN(video.duration)) ? video.duration : 1;

    const effectiveSeconds = watchTime * 60;
    const effectiveRatio = clamp((effectiveSeconds / duration) * 100);

    // 原始行为归一化
    const watchNorm = normalizeCount(watchTime, 15);
    const askNorm = normalizeCount(askCount, 6);
    const pauseNorm = normalizeCount(pauseCount, 6);
    const rewatchNorm = normalizeCount(rewatchCount, 5);
    const seekNorm = normalizeCount(seekCount, 6);

    // 1. 学习投入度
    const engagement = clamp(
        watchNorm * 0.30 +
        progress * 0.30 +
        effectiveRatio * 0.25 +
        askNorm * 0.15
    );

    // 2. 理解压力
    const pressure = clamp(
        pauseNorm * 0.35 +
        rewatchNorm * 0.35 +
        askNorm * 0.30
    );

    // 3. 学习连续性
    const progressEffectiveGap = Math.max(0, progress - effectiveRatio);
    const continuity = clamp(
        100 -
        seekNorm * 0.55 -
        progressEffectiveGap * 0.35 -
        pauseNorm * 0.10
    );

    return {
        raw: { watchTime, askCount, pauseCount, rewatchCount, seekCount, progress, effectiveRatio },
        vector: [engagement, pressure, continuity],
        summary: { engagement, pressure, continuity }
    };
}

// ============================================================
// K-means 核心算法
// ============================================================
function weightedEuclidean(p1, p2) {
    let sum = 0;
    for (let i = 0; i < p1.length; i++) {
        const diff = (p1[i] - p2[i]) * FEATURE_WEIGHTS[i];
        sum += diff * diff;
    }
    return Math.sqrt(sum);
}

function assignCluster(point, centroids) {
    let minDist = Infinity, bestIdx = 0;
    centroids.forEach((c, i) => {
        const d = weightedEuclidean(point, c);
        if (d < minDist) { minDist = d; bestIdx = i; }
    });
    return bestIdx;
}

function kmeansIterate(points, initCentroids, maxIter = MAX_ITER) {
    if (points.length < K) {
        return {
            centroids: initCentroids.map(c => [...c]),
            assignments: points.map(p => assignCluster(p, initCentroids)),
            iterations: 0,
            converged: false
        };
    }

    let centroids = initCentroids.map(c => [...c]);
    let assignments = new Array(points.length).fill(0);
    let iter = 0;

    for (iter = 0; iter < maxIter; iter++) {
        const newAssignments = points.map(p => assignCluster(p, centroids));

        const newCentroids = centroids.map(c => [...c]);
        for (let k = 0; k < K; k++) {
            const members = points.filter((_, i) => newAssignments[i] === k);
            if (members.length > 0) {
                for (let d = 0; d < centroids[0].length; d++) {
                    newCentroids[k][d] = members.reduce((s, p) => s + p[d], 0) / members.length;
                }
            }
        }

        let maxShift = 0;
        for (let k = 0; k < K; k++) {
            const shift = weightedEuclidean(centroids[k], newCentroids[k]);
            if (shift > maxShift) maxShift = shift;
        }

        centroids = newCentroids;
        assignments = newAssignments;

        if (maxShift < CONVERGE_THRESHOLD) { iter++; break; }
    }

    return { centroids, assignments, iterations: iter, converged: iter < maxIter };
}

// 填充历史数据（三维）
function seedHistoricalData() {
    const seeds = [
        // 新晋体验者群
        [20, 15, 80], [30, 25, 70], [22, 18, 78], [28, 20, 72], [25, 22, 75],
        // 基础薄弱型群
        [50, 75, 55], [60, 85, 65], [55, 80, 58], [52, 78, 62], [58, 82, 60],
        // 稳步提升型群
        [70, 40, 82], [78, 50, 78], [72, 42, 85], [76, 48, 80], [74, 45, 83],
        // 强化钻研型群
        [88, 30, 92], [92, 38, 88], [90, 32, 90], [85, 35, 95], [93, 36, 87],
        // 碎片跳学型群
        [42, 38, 28], [48, 42, 32], [44, 35, 25], [46, 40, 35], [50, 45, 30]
    ];
    dataPoints = seeds.map(s => [...s]);
}

function runKMeans(features) {
    const userVector = features.vector;

    if (dataPoints._lastUserIdx !== undefined) {
        dataPoints[dataPoints._lastUserIdx] = userVector;
    } else {
        dataPoints.push(userVector);
        dataPoints._lastUserIdx = dataPoints.length - 1;
    }

    const result = kmeansIterate(dataPoints, currentCentroids);
    currentCentroids = result.centroids;

    iterationLog.push({
        time: Date.now(),
        iterations: result.iterations,
        converged: result.converged,
        centroids: result.centroids.map(c => [...c])
    });

    const userCluster = assignCluster(userVector, currentCentroids);

    const distances = currentCentroids.map((c, i) => ({
        centroid: { ...PROFILE_META[i], center: c },
        dist: weightedEuclidean(userVector, c)
    }));

    const profile = { ...PROFILE_META[userCluster], center: currentCentroids[userCluster] };
    profile.suggestion = generateSimpleSuggestion(profile, features.summary, features.raw);

    return { profile, distances, kmeansInfo: result };
}

function getCurrentCentroids() {
    return currentCentroids.map((c, i) => ({ ...PROFILE_META[i], center: c }));
}

// 初始化历史数据
seedHistoricalData();

// ============================================================
// 个性化建议
// ============================================================
function generateSimpleSuggestion(profile, summary, raw) {
    const tips = [profile.suggestion];

    if (summary.pressure > 70) {
        tips.push("系统检测到你的理解压力偏高，建议优先复习本节核心概念。");
    }
    if (summary.continuity < 45) {
        tips.push("你的学习连续性偏低，建议减少频繁跳转，按知识顺序完成学习。");
    }
    if (summary.engagement > 80 && summary.pressure < 50) {
        tips.push("当前学习状态较好，可以尝试挑战进阶题目。");
    }
    if (raw.progress > 80 && raw.effectiveRatio < 50) {
        tips.push("播放进度较高但有效观看比例偏低，建议重新观看关键片段。");
    }

    return tips.join(" ");
}

// ============================================================
// 主入口
// ============================================================
function runMLModel() {
    const features = buildLearningFeatures();
    const time = features.raw.watchTime;
    const speak = features.raw.askCount;

    document.getElementById('statTime').textContent = time.toFixed(1);
    document.getElementById('statSpeak').textContent = speak;

    const el = (id) => document.getElementById(id);
    if (el('statPause')) el('statPause').textContent = pauseCount;
    if (el('statRewatch')) el('statRewatch').textContent = rewatchCount;
    if (el('statSeek')) el('statSeek').textContent = seekCount;
    if (el('statEffective')) el('statEffective').textContent = Math.round(features.raw.effectiveRatio) + '%';
    if (el('statEngagement')) el('statEngagement').textContent = Math.round(features.summary.engagement);
    if (el('statPressure')) el('statPressure').textContent = Math.round(features.summary.pressure);
    if (el('statContinuity')) el('statContinuity').textContent = Math.round(features.summary.continuity);

    const result = runKMeans(features);

    drawKMeansCanvas(features.summary.engagement, features.summary.pressure, result.profile);
    drawRadarCanvas(result.profile);
    renderProfile(result.profile);
    renderDistanceTable(result.distances, result.profile);
    trackProfileChange(features, result.profile);
}

function renderProfile(p) {
    document.getElementById('profileName').textContent = p.name;
    document.getElementById('profileName').style.color = p.color;
    document.getElementById('profileScore').textContent = p.score;
    document.getElementById('profileScore').style.color = p.color;
    document.getElementById('profileSuggestion').textContent = p.suggestion;
    document.getElementById('profileCard').style.borderLeftColor = p.color;
    document.getElementById('profileTags').innerHTML = p.tags.map(t =>
        `<span class="tag" style="background:${p.color}22;color:${p.color};">#${t}</span>`
    ).join('');
}

function renderDistanceTable(distances, best) {
    const maxDist = Math.max(...distances.map(d => d.dist)) || 1;
    document.getElementById('distanceTbody').innerHTML = distances.map(({ centroid, dist }) => {
        const isBest = centroid.id === best.id;
        const barW = Math.max(4, Math.round((1 - dist / maxDist) * 60));
        return `<tr class="${isBest ? 'best-match' : ''}">
            <td style="color:${centroid.color}">${isBest ? '★ ' : ''}${centroid.name}</td>
            <td>${dist.toFixed(2)}</td>
            <td><div class="dist-bar-wrap"><div class="dist-bar" style="width:${barW}px;background:${centroid.color};"></div></div></td>
        </tr>`;
    }).join('');
}

// ============================================================
// K-means 散点图（二维投影：学习投入度 × 理解压力）
// ============================================================
function drawKMeansCanvas(engagementVal, pressureVal, bestProfile) {
    const canvas = document.getElementById('kmeansCanvas');
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const PAD = 36;
    const maxX = 100, maxY = 100;

    const toPx = (x, y) => [
        PAD + (x / maxX) * (W - PAD * 2),
        H - PAD - (Math.min(y, maxY) / maxY) * (H - PAD * 2)
    ];

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#12141f';
    ctx.fillRect(0, 0, W, H);

    // 网格
    ctx.strokeStyle = '#1e2030';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
        const x = PAD + (i / 4) * (W - PAD * 2);
        ctx.beginPath(); ctx.moveTo(x, PAD); ctx.lineTo(x, H - PAD); ctx.stroke();
        const y = PAD + (i / 4) * (H - PAD * 2);
        ctx.beginPath(); ctx.moveTo(PAD, y); ctx.lineTo(W - PAD, y); ctx.stroke();
    }

    // 坐标轴
    ctx.strokeStyle = '#3a3d50';
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(PAD, PAD); ctx.lineTo(PAD, H - PAD); ctx.lineTo(W - PAD, H - PAD); ctx.stroke();

    // 轴标签
    ctx.fillStyle = '#666';
    ctx.font = '10px Microsoft YaHei';
    ctx.textAlign = 'center';
    ctx.fillText('学习投入度', W / 2, H - 8);
    ctx.save();
    ctx.translate(12, H / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('理解压力', 0, 0);
    ctx.restore();

    // 刻度
    ctx.fillStyle = '#555';
    ctx.font = '9px Arial';
    for (let i = 0; i <= 4; i++) {
        const xVal = (i / 4) * maxX;
        ctx.textAlign = 'center';
        ctx.fillText(xVal.toFixed(0), toPx(xVal, 0)[0], H - PAD + 12);
        const yVal = (i / 4) * maxY;
        ctx.textAlign = 'right';
        ctx.fillText(yVal.toFixed(0), PAD - 4, toPx(0, yVal)[1] + 3);
    }

    // 历史数据点（投影到前两维）
    dataPoints.forEach((p, idx) => {
        if (idx === dataPoints._lastUserIdx) return;
        const [px, py] = toPx(p[0], p[1]);
        const cluster = assignCluster(p, currentCentroids);
        ctx.fillStyle = PROFILE_META[cluster].color + '66';
        ctx.beginPath(); ctx.arc(px, py, 3, 0, Math.PI * 2); ctx.fill();
    });

    // 用户到最佳质心虚线
    const CENTROIDS_NOW = getCurrentCentroids();
    if (engagementVal > 0 || pressureVal > 0) {
        const [ux, uy] = toPx(engagementVal, pressureVal);
        const [bx, by] = toPx(bestProfile.center[0], bestProfile.center[1]);
        ctx.strokeStyle = bestProfile.color;
        ctx.setLineDash([4, 3]);
        ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(ux, uy); ctx.lineTo(bx, by); ctx.stroke();
        ctx.setLineDash([]);
    }

    // 质心
    CENTROIDS_NOW.forEach(c => {
        const [cx, cy] = toPx(c.center[0], c.center[1]);
        const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, 30);
        grad.addColorStop(0, c.color + '55');
        grad.addColorStop(1, c.color + '00');
        ctx.fillStyle = grad;
        ctx.beginPath(); ctx.arc(cx, cy, 30, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = c.color;
        ctx.beginPath(); ctx.arc(cx, cy, 6, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.fillStyle = c.color;
        ctx.font = 'bold 10px Microsoft YaHei';
        ctx.textAlign = 'center';
        ctx.fillText(c.name, cx, cy - 12);
    });

    // 用户当前点
    if (engagementVal > 0 || pressureVal > 0) {
        const [ux, uy] = toPx(engagementVal, pressureVal);
        const pulse = Math.sin(Date.now() / 300) * 3 + 10;
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(ux, uy, pulse, 0, Math.PI * 2); ctx.stroke();
        ctx.fillStyle = '#fff';
        ctx.beginPath(); ctx.arc(ux, uy, 5, 0, Math.PI * 2); ctx.fill();
        ctx.font = 'bold 10px Arial';
        ctx.fillText('YOU', ux, uy - 15);
    }

    // 迭代信息
    const lastLog = iterationLog[iterationLog.length - 1];
    if (lastLog) {
        ctx.fillStyle = '#2ecc71';
        ctx.font = 'bold 10px Arial';
        ctx.textAlign = 'right';
        ctx.fillText(`✓ K=${K} 迭代${lastLog.iterations}次 | ${lastLog.converged ? '已收敛' : '运行中'}`, W - 8, 14);
    }
}

// ============================================================
// 雷达图
// ============================================================
function drawRadarCanvas(profile) {
    const canvas = document.getElementById('radarCanvas');
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const cx = W / 2, cy = H / 2;
    const R = 70;
    const N = RADAR_LABELS.length;

    ctx.clearRect(0, 0, W, H);

    ctx.strokeStyle = '#2a2d3e';
    ctx.lineWidth = 1;
    for (let r = 1; r <= 4; r++) {
        ctx.beginPath();
        for (let i = 0; i < N; i++) {
            const angle = (Math.PI * 2 / N) * i - Math.PI / 2;
            const x = cx + Math.cos(angle) * (R / 4) * r;
            const y = cy + Math.sin(angle) * (R / 4) * r;
            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.closePath();
        ctx.stroke();
    }

    for (let i = 0; i < N; i++) {
        const angle = (Math.PI * 2 / N) * i - Math.PI / 2;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(cx + Math.cos(angle) * R, cy + Math.sin(angle) * R);
        ctx.stroke();
    }

    ctx.fillStyle = '#888';
    ctx.font = '10px Microsoft YaHei';
    ctx.textAlign = 'center';
    for (let i = 0; i < N; i++) {
        const angle = (Math.PI * 2 / N) * i - Math.PI / 2;
        const x = cx + Math.cos(angle) * (R + 14);
        const y = cy + Math.sin(angle) * (R + 14) + 3;
        ctx.fillText(RADAR_LABELS[i], x, y);
    }

    ctx.fillStyle = profile.color + '55';
    ctx.strokeStyle = profile.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < N; i++) {
        const angle = (Math.PI * 2 / N) * i - Math.PI / 2;
        const v = profile.radar[i] / 100;
        const x = cx + Math.cos(angle) * R * v;
        const y = cy + Math.sin(angle) * R * v;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    for (let i = 0; i < N; i++) {
        const angle = (Math.PI * 2 / N) * i - Math.PI / 2;
        const v = profile.radar[i] / 100;
        const x = cx + Math.cos(angle) * R * v;
        const y = cy + Math.sin(angle) * R * v;
        ctx.fillStyle = profile.color;
        ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI * 2); ctx.fill();
    }
}

// ============================================================
// 视频监听
// ============================================================
function bindVideoEvents() {
    const video = document.getElementById('myVideo');
    const inputTime = document.getElementById('inputTime');
    const progFill = document.getElementById('progressFill');
    const progText = document.getElementById('progressText');
    if (!video) return;

    let accumulatedSeconds = (parseFloat(inputTime.value) || 0) * 60;
    let lastTimestamp = 0;
    let lastModelTrigger = -1;

    video.addEventListener('timeupdate', () => {
        const now = video.currentTime;
        const delta = now - lastTimestamp;

        if (delta > 0 && delta < 2) {
            accumulatedSeconds += delta;
        }

        if (delta < -1) {
            rewatchCount++;
            seekCount++;
        }

        if (delta > 2) {
            seekCount++;
        }

        lastTimestamp = now;

        const minutes = (accumulatedSeconds / 60).toFixed(2);
        inputTime.value = minutes;

        if (video.duration) {
            const pct = (now / video.duration * 100).toFixed(1);
            progFill.style.width = pct + '%';
            progText.textContent = pct + '%';
        }

        const triggerKey = Math.floor(accumulatedSeconds / 3);
        if (triggerKey !== lastModelTrigger) {
            lastModelTrigger = triggerKey;
            runMLModel();
        }
    });

    video.addEventListener('pause', () => {
        pauseCount++;
        runMLModel();
        saveLearningLog();
    });

    video.addEventListener('ended', () => {
        runMLModel();
        saveLearningLog();
    });
}

// ============================================================


async function loadLatestLearningState() {
    try {
        const res = await fetch(`${BACKEND_URL}/api/learning/latest?userId=${currentUserId}&courseId=${currentCourseId}`);
        const data = await res.json();
        if (data.code !== 0 || !data.data) return;
        const r = data.data;
        document.getElementById('inputTime').value = Number(r.watch_time || 0).toFixed(2);
        document.getElementById('inputSpeak').value = Number(r.ask_count || 0);
        chatCount = Number(r.ask_count || 0);
        pauseCount = Number(r.pause_count || 0);
        rewatchCount = Number(r.rewatch_count || 0);
        seekCount = Number(r.seek_count || 0);
        console.log('已恢复上次学习状态');
    } catch (e) {
        console.warn('加载学习状态失败:', e);
    }
}



// 保存学习记录
// ============================================================
async function saveLearningLog() {
    try {
        const features = buildLearningFeatures();
        const payload = {
            userId: currentUserId,
            courseId: currentCourseId,
            watchTime: features.raw.watchTime,
            askCount: features.raw.askCount,
            pauseCount: features.raw.pauseCount,
            rewatchCount: features.raw.rewatchCount,
            seekCount: features.raw.seekCount,
            progress: features.raw.progress,
            effectiveRatio: features.raw.effectiveRatio,
            engagement: features.summary.engagement,
            pressure: features.summary.pressure,
            continuity: features.summary.continuity,
            profileName: document.getElementById('profileName')?.textContent || '未知',
            score: parseInt(document.getElementById('profileScore')?.textContent) || 0
        };

        await fetch(`${BACKEND_URL}/api/learning/log`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
    } catch (e) {
        console.error('保存学习记录失败:', e);
    }
}


// ============================================================
// 画像变化追踪
// ============================================================
let lastProfileName = '';
let profileTimelineLocal = [];

async function trackProfileChange(features, profile) {
    // 只在画像发生变化时记录
    if (profile.name === lastProfileName) return;
    lastProfileName = profile.name;

    const record = {
        time: new Date().toLocaleTimeString(),
        name: profile.name,
        score: profile.score,
        color: profile.color,
        engagement: Math.round(features.summary.engagement),
        pressure: Math.round(features.summary.pressure),
        continuity: Math.round(features.summary.continuity),
        watchTime: features.raw.watchTime,
        askCount: features.raw.askCount
    };

    // 本地记录
    profileTimelineLocal.push(record);
    renderTimeline();

    // 后端保存
    try {
        await fetch(`${BACKEND_URL}/api/profile/save`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                userId: currentUserId,
                courseId: currentCourseId,
                profileName: record.name,
                score: record.score,
                engagement: record.engagement,
                pressure: record.pressure,
                continuity: record.continuity,
                watchTime: record.watchTime,
                askCount: record.askCount
            })
        });
    } catch (e) {
        console.warn('保存画像历史失败:', e);
    }
}

function renderTimeline() {
    const container = document.getElementById('profileTimeline');
    if (!container) return;

    if (profileTimelineLocal.length === 0) {
        container.innerHTML = '<p style="color:#555;">开始学习后自动记录画像变化...</p>';
        return;
    }

    container.innerHTML = profileTimelineLocal.map((r, i) => `
        <div style="display:flex;align-items:center;gap:8px;padding:6px 0;border-bottom:1px solid #1e2030;">
            <div style="width:8px;height:8px;border-radius:50%;background:${r.color};flex-shrink:0;"></div>
            <div style="flex:1;">
                <span style="color:${r.color};font-weight:bold;">${r.name}</span>
                <span style="color:#666;margin-left:6px;">${r.score}分</span>
                <span style="color:#555;margin-left:6px;font-size:11px;">
                    投入${r.engagement} | 压力${r.pressure} | 连续${r.continuity}
                </span>
            </div>
            <div style="color:#444;font-size:10px;flex-shrink:0;">${r.time}</div>
        </div>
    `).join('');

    container.scrollTop = container.scrollHeight;
}

// 页面加载时拉取历史记录
async function loadProfileHistory() {
    try {
        const res = await fetch(`${BACKEND_URL}/api/profile/history?userId=${currentUserId}&courseId=${currentCourseId}`);
        const data = await res.json();

        if (data.code === 0 && data.data.length > 0) {
            profileTimelineLocal = data.data.map(r => ({
                time: new Date(r.created_at).toLocaleTimeString(),
                name: r.profile_name,
                score: r.score,
                color: (PROFILE_META.find(p => p.name === r.profile_name) || {}).color || '#888',
                engagement: Math.round(r.engagement),
                pressure: Math.round(r.pressure),
                continuity: Math.round(r.continuity),
                watchTime: r.watch_time,
                askCount: r.ask_count
            }));

            if (profileTimelineLocal.length > 0) {
                lastProfileName = profileTimelineLocal[profileTimelineLocal.length - 1].name;
            }

            renderTimeline();
        }
    } catch (e) {
        console.warn('加载画像历史失败:', e);
    }
}


// ============================================================
// 动画循环
// ============================================================
function animationLoop() {
    const features = buildLearningFeatures();
    const result = runKMeans(features);
    drawKMeansCanvas(features.summary.engagement, features.summary.pressure, result.profile);
    requestAnimationFrame(animationLoop);
}

// ============================================================
// AI 聊天
// ============================================================
async function askAI() {
    const input = document.getElementById('userInput');
    const box = document.getElementById('chatBox');
    const question = input.value.trim();
    if (!question) return;

    box.innerHTML += `<div style="color:#c0c4ff;margin-bottom:8px;"><strong>你：</strong>${question}</div>`;
    input.value = '';

    chatCount++;
    document.getElementById('inputSpeak').value = chatCount;
    runMLModel();

    const loading = document.createElement('div');
    loading.style.color = '#7c83f5';
    loading.style.marginBottom = '8px';
    loading.innerHTML = `<strong>AI：</strong><i class="fas fa-spinner fa-spin"></i> 思考中...`;
    box.appendChild(loading);
    box.scrollTop = box.scrollHeight;

    try {
        const res = await fetch(`${BACKEND_URL}/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question })
        });

        const data = await res.json();
        if (box.contains(loading)) box.removeChild(loading);

        if (!res.ok || data.code !== 0) {
            box.innerHTML += `<div style="color:#e74c3c;">AI 回复失败：${data.message || '未知错误'}</div>`;
            box.scrollTop = box.scrollHeight;
            return;
        }

        const answer = data.data?.answer || '暂无回复';
        box.innerHTML += `<div style="color:#7c83f5;margin-bottom:8px;"><strong>AI：</strong>${answer}</div>`;
        box.scrollTop = box.scrollHeight;
    } catch (e) {
        if (box.contains(loading)) box.removeChild(loading);
        box.innerHTML += `<div style="color:#e74c3c;">网络错误，请稍后再试。</div>`;
               box.scrollTop = box.scrollHeight;
    }
}

// ============================================================
// AI 出题
// ============================================================
async function generateQuizWithAI() {
    const quizContent = document.getElementById('quizContent');
    const btn = document.getElementById('generateQuizBtn');
    const resultDiv = document.getElementById('quizResult');

    resultDiv.style.display = 'none';
    resultDiv.innerHTML = '';

    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> AI 正在构思题目...';

    try {
        const features = buildLearningFeatures();

        const res = await fetch(`${BACKEND_URL}/api/quiz/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                userId: currentUserId,
                courseId: currentCourseId,
                profileName: document.getElementById('profileName')?.textContent || '普通学习者',
                behavior: features.raw
            })
        });

        const data = await res.json();
        if (!res.ok || data.code !== 0) {
            quizContent.innerHTML = `<p style="color:#e74c3c;">生成测验失败：${data.message || '未知错误'}</p>`;
            return;
        }

        const { quizId, question, options } = data.data;
        currentQuizId = quizId;

        quizContent.innerHTML = `
            <div id="dynamicQuiz">
                <p class="quiz-q">${question}</p>
                ${options.map((opt, i) => `
                    <label style="display:block;margin:8px 0;">
                        <input type="radio" name="ai_quiz" value="${i}">
                        ${opt}
                    </label>
                `).join('')}
                <button class="btn" onclick="checkAIDynamicAnswer()" style="margin-top:15px;width:100%;justify-content:center;">
                    <i class="fas fa-check"></i> 确认提交
                </button>
            </div>
        `;
    } catch (e) {
        quizContent.innerHTML = `<p style="color:#e74c3c;">网络错误，生成失败</p>`;
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-magic"></i> 生成 AI 练习题';
    }
}

async function checkAIDynamicAnswer() {
    const resultDiv = document.getElementById('quizResult');
    const selected = document.querySelector('input[name="ai_quiz"]:checked');

    if (!currentQuizId) {
        alert('请先生成题目');
        return;
    }
    if (!selected) {
        alert('请先选择答案');
        return;
    }

    const userAnswer = Number(selected.value);

    resultDiv.style.display = 'block';
    resultDiv.innerHTML = '提交中...';

    try {
        const res = await fetch(`${BACKEND_URL}/api/quiz/submit`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                quizId: currentQuizId,
                userAnswer
            })
        });

        const data = await res.json();
        if (!res.ok || data.code !== 0) {
            resultDiv.innerHTML = `<strong style="color:#e74c3c;">提交失败：${data.message || '未知错误'}</strong>`;
            return;
        }

        resultDiv.innerHTML = `
            <strong>${data.data.isCorrect ? '✅ 回答正确' : '❌ 回答错误'}</strong>
            <hr style="border-color:#2a2d3e;margin:8px 0;">
            正确答案索引：${data.data.correctIndex}
            <br>
            解析：${data.data.explanation || '暂无解析'}
        `;
    } catch (e) {
        resultDiv.innerHTML = `<strong style="color:#e74c3c;">网络错误，请稍后再试</strong>`;
    }
}


// ============================================================
// 数据导出
// ============================================================
function exportAllData() {
    const data = {
        profileTimeline: profileTimelineLocal,
        exportTime: new Date().toISOString()
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `learning_data_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
}



// ============================================================
// 初始化
// ============================================================
window.addEventListener('load', async () => {
    await loadLatestLearningState();
    await loadProfileHistory();
    bindVideoEvents();
    runMLModel();
    animationLoop();
    setInterval(() => { saveLearningLog(); }, 30000);
    document.getElementById('sendBtn').onclick = askAI;
    document.getElementById('userInput').onkeypress = (e) => { if (e.key === 'Enter') askAI(); };
});