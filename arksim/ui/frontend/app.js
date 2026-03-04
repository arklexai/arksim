/* Arksim Control Plane — Alpine.js application */

function arksim() {
  return {
    // ── Navigation ──────────────────────────────
    activePage: 'build',
    get isRunning() {
      return this.simStatus === 'running' || this.evalStatus === 'running';
    },

    navItems: [
      { id: 'build',    label: 'Build',    caption: 'Create scenarios',     icon: 'build',       step: 0 },
      { id: 'simulate', label: 'Simulate', caption: 'Generate conversations', icon: 'play_circle', step: 1 },
      { id: 'evaluate', label: 'Evaluate', caption: 'Score conversations', icon: 'fact_check',  step: 2 },
      { id: 'results',  label: 'Results',  caption: 'View scores & report', icon: 'assessment',  step: 3 },
    ],

    navigate(page) {
      if (this.isRunning && page !== this.activePage) return;
      this.activePage = page;
    },

    // ── Page Color Scheme ──────────────────────
    _pageColors: {
      simulate: { light: '#2563eb', dark: '#60a5fa' },
      evaluate: { light: '#7c3aed', dark: '#a78bfa' },
      results:  { light: '#059669', dark: '#34d399' },
      build:    { light: '#d97706', dark: '#fbbf24' },
    },

    _navColor(pageId) {
      const c = this._pageColors[pageId] || this._pageColors.simulate;
      return this.darkMode ? c.dark : c.light;
    },

    buildDone: false,

    stepStatus(pageId) {
      if (pageId === 'build') return this.buildDone ? 'done' : 'idle';
      if (pageId === 'simulate') return this.simStatus;
      if (pageId === 'evaluate') return this.evalStatus;
      if (pageId === 'results') return this.evalResults ? 'done' : 'idle';
      return 'idle';
    },

    // ── Defaults ──────────────────────────────
    _defaults: {
      model: 'gpt-5.1',
      provider: 'openai',
      outputFilePath: '',
      numConversations: 5,
      maxTurns: 5,
      numWorkers: 'auto',
      evalNumWorkers: 'auto',
      generateHtmlReport: true,
    },

    // ── Form State ──────────────────────────────
    agentConfigFilePath: '',
    model: 'gpt-5.1',
    provider: 'openai',
    numConversations: 5,
    maxTurns: 5,
    numWorkers: 'auto',
    outputFilePath: '',

    evalAgentConfigFilePath: '',
    evalModel: 'gpt-5.1',
    evalProvider: 'openai',
    evalInputSource: 'auto',
    evalSimulationFilePath: '',
    evalNumWorkers: 'auto',
    generateHtmlReport: true,
    scoreThreshold: '',
    metricsToRun: [],
    customMetricsFilePaths: '',

    resultsInputSource: 'auto',
    resultsInputDir: '',

    // ── Scenario Builder ─────────────────────────
    scenarios: [],
    scenarioFilePath: '',
    scenarioDirty: false,
    scenarioIsDemo: false,
    scenarioFileValid: false,

    // ── Model / Provider Mapping ────────────────
    providerModels: {
      openai:  ['gpt-5.1', 'gpt-4.1', 'gpt-4.1-mini', 'o3', 'o4-mini'],
      azure:   ['gpt-5.1', 'gpt-4.1', 'gpt-4.1-mini'],
      claude:  ['claude-sonnet-4-20250514', 'claude-haiku-4-5-20251001'],
      gemini:  ['gemini-2.5-flash', 'gemini-2.5-pro'],
    },
    providers: ['openai', 'azure', 'claude', 'gemini'],

    modelsForProvider(provider) {
      return this.providerModels[provider] || [];
    },

    providerForModel(model) {
      for (const [prov, models] of Object.entries(this.providerModels)) {
        if (models.includes(model)) return prov;
      }
      return null;
    },

    isKnownModel(model) {
      for (const models of Object.values(this.providerModels)) {
        if (models.includes(model)) return true;
      }
      return false;
    },

    providerEnvHints: {
      openai:  'OPENAI_API_KEY',
      azure:   'AZURE_CLIENT_ID, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_VERSION',
      claude:  'ANTHROPIC_API_KEY',
      gemini:  'GEMINI_API_KEY',
    },

    envHint(provider) {
      return this.providerEnvHints[provider] || null;
    },

    // ── Job State ───────────────────────────────
    simStatus: 'idle',
    simOutputDir: null,
    simError: null,
    simProgress: 0,
    evalStatus: 'idle',
    evalOutputDir: null,
    evalError: null,
    evalProgress: 0,
    evalResults: null,

    // ── Logs ────────────────────────────────────
    logs: [],
    showLogs: false,

    // ── Config ──────────────────────────────────
    projectRoot: '',
    configs: [],
    selectedConfig: '',
    _loadedConfig: null,  // full settings object from loaded YAML

    // ── Directory/File Browser ────────────────────
    showBrowser: false,
    browserPath: '',
    browserEntries: [],
    browserParent: null,
    browserMode: 'directory', // 'directory' | 'file'
    browserFileFilter: null,  // e.g. ['.yaml', '.yml']
    _browserResolve: null,

    // ── Dark Mode ─────────────────────────────────
    darkMode: false,

    _initDarkMode() {
      this.darkMode = localStorage.getItem('arksim-dark') === 'true';
      document.documentElement.classList.toggle('dark', this.darkMode);
    },

    toggleDark() {
      this.darkMode = !this.darkMode;
      localStorage.setItem('arksim-dark', this.darkMode);
      document.documentElement.classList.toggle('dark', this.darkMode);
    },

    // ── WebSocket ───────────────────────────────
    _ws: null,

    // ── Init ────────────────────────────────────
    async init() {
      this._initDarkMode();
      this._connectWs();
      await this._fetchProjectRoot();
      await this._loadConfigs();
      if (this.scenarios.length === 0) await this._loadDemoScenario();
    },

    async _fetchProjectRoot() {
      try {
        const resp = await fetch('/api/fs/root');
        const data = await resp.json();
        this.projectRoot = data.root || '';
      } catch { /* ignore */ }
    },

    // ── WebSocket ───────────────────────────────
    _connectWs() {
      if (this._ws) {
        this._ws.onclose = null;
        this._ws.close();
      }
      const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
      this._ws = new WebSocket(`${proto}//${location.host}/api/ws/logs`);
      this._ws.onmessage = (e) => {
        const msg = JSON.parse(e.data);
        if (msg.type === 'log') {
          this.logs = [...this.logs, msg];
          this.$nextTick(() => {
            const el = document.getElementById('log-container');
            if (el) el.scrollTop = el.scrollHeight;
          });
        } else if (msg.type === 'progress') {
          this._handleProgress(msg);
        } else if (msg.type === 'status') {
          this._handleStatus(msg);
        }
      };
      this._ws.onclose = () => setTimeout(() => this._connectWs(), 3000);
    },

    _handleProgress(msg) {
      const pct = msg.total > 0 ? Math.round((msg.completed / msg.total) * 100) : 0;
      // Cap at 95% — post-processing (saving, reports) still
      // runs after turns complete. 100% is set by _handleStatus
      // when the job actually finishes.
      const capped = Math.min(pct, 95);
      // Only move forward — out-of-order messages from the
      // background thread can arrive with lower values.
      if (msg.job === 'simulate' && capped > this.simProgress) this.simProgress = capped;
      else if (msg.job === 'evaluate' && capped > this.evalProgress) this.evalProgress = capped;
    },

    _handleStatus(msg) {
      if (msg.job === 'simulate') {
        this.simStatus = msg.status;
        if (msg.status === 'done') this.simProgress = 100;
        if (msg.output_dir) this.simOutputDir = msg.output_dir;
        if (msg.error) this.simError = msg.error;
      } else if (msg.job === 'evaluate') {
        this.evalStatus = msg.status;
        if (msg.status === 'done') this.evalProgress = 100;
        if (msg.output_dir) this.evalOutputDir = msg.output_dir;
        if (msg.error) this.evalError = msg.error;
        if (msg.status === 'done') this._fetchResults();
      }
    },

    // ── API Calls ───────────────────────────────
    async cancelSimulation() {
      await fetch('/api/simulate/cancel', { method: 'POST' });
    },

    async cancelEvaluation() {
      await fetch('/api/evaluate/cancel', { method: 'POST' });
    },

    async startSimulation() {
      if (this.simStatus === 'running') return;
      this.logs = [];
      this.simError = null;
      this.simProgress = 0;
      this.simStatus = 'running';
      this.showLogs = true;

      const d = this._defaults;
      const resp = await fetch('/api/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          agent_config_file_path: this.agentConfigFilePath || null,
          agent_config: this._loadedConfig?.agent_config || null,
          model: this.model || d.model,
          provider: this.provider || d.provider,
          num_conversations: parseInt(this.numConversations) || d.numConversations,
          max_turns: parseInt(this.maxTurns) || d.maxTurns,
          num_workers: (this.numWorkers || d.numWorkers) === 'auto' ? 'auto' : parseInt(this.numWorkers),
          scenario_file: this.scenarioFilePath || null,
          output_file_path: this.outputFilePath || null,
        }),
      });
      const data = await resp.json();
      if (!resp.ok || data.error) {
        this.simStatus = 'failed';
        this.simError = data.error || JSON.stringify(data.detail ?? data);
      }
    },

    async startEvaluation() {
      if (this.evalStatus === 'running') return;
      this.logs = [];
      this.evalError = null;
      this.evalProgress = 0;
      this.evalStatus = 'running';
      this.showLogs = true;

      const simulationFilePath = this.evalInputSource === 'auto'
        ? (this.simOutputDir ? `${this.simOutputDir}/simulation.json` : null)
        : this.evalSimulationFilePath || null;

      const d = this._defaults;
      const resp = await fetch('/api/evaluate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          simulation_file_path: simulationFilePath,
          scenario_file_path: this.scenarioFilePath || null,
          model: this.evalModel || d.model,
          provider: this.evalProvider || d.provider,
          num_workers: (this.evalNumWorkers || d.evalNumWorkers) === 'auto' ? 'auto' : parseInt(this.evalNumWorkers),
          generate_html_report: this.generateHtmlReport ?? d.generateHtmlReport,
          score_threshold: this.scoreThreshold ? parseFloat(this.scoreThreshold) : null,
          metrics_to_run: this.metricsToRun.length > 0 ? this.metricsToRun : null,
          custom_metrics_file_paths: this.customMetricsFilePaths
            ? this.customMetricsFilePaths.split(',').map(s => s.trim()).filter(Boolean)
            : [],
          output_file_path: this.outputFilePath || null,
        }),
      });
      const data = await resp.json();
      if (!resp.ok || data.error) {
        this.evalStatus = 'failed';
        this.evalError = data.error || JSON.stringify(data.detail ?? data);
      }
    },

    async _fetchResults() {
      const dir = this.resultsInputSource === 'auto'
        ? this.evalOutputDir
        : this.resultsInputDir;
      if (!dir) return;
      const resp = await fetch(`/api/results?dir=${encodeURIComponent(dir)}`);
      const data = await resp.json();
      if (data.results) this.evalResults = data.results;
    },

    async refreshResults() {
      await this._fetchResults();
    },

    // ── Config ──────────────────────────────────
    async _loadConfigs() {
      const resp = await fetch('/api/fs/configs');
      const data = await resp.json();
      this.configs = (data.configs || []).filter(c =>
        !c.relative.includes('evaluat')
      );
      if (this.configs.length > 0) await this._applyConfig(this.configs[0].path);
    },

    async _applyConfig(path) {
      if (!path) return;
      this.selectedConfig = path;
      const resp = await fetch(`/api/fs/config?path=${encodeURIComponent(path)}`);
      const data = await resp.json();
      if (!data.settings) return;
      const s = data.settings;
      const d = this._defaults;
      this._loadedConfig = s;
      this.agentConfigFilePath = s.agent_config_file_path || '';
      this.model = s.model || d.model;
      this.provider = s.provider || d.provider;
      const _convs = s.num_conversations || (s.num_conversations_per_scenario > 1 ? s.num_conversations_per_scenario : 0);
      this.numConversations = _convs || d.numConversations;
      this.maxTurns = s.max_turns || d.maxTurns;
      this.numWorkers = s.num_workers !== undefined ? String(s.num_workers) : d.numWorkers;
      this.outputFilePath = path ? path.substring(0, path.lastIndexOf('/')) : '';
      if (s.scenario_file_path) {
        this.scenarioFilePath = s.scenario_file_path;
        await this.loadScenarioFile();
      }
      this.generateHtmlReport = s.generate_html_report !== undefined ? s.generate_html_report : d.generateHtmlReport;
      this.scoreThreshold = s.score_threshold != null ? String(s.score_threshold) : '';
      this.metricsToRun = Array.isArray(s.metrics_to_run) ? [...s.metrics_to_run] : [];
      this.customMetricsFilePaths = Array.isArray(s.custom_metrics_file_paths)
        ? s.custom_metrics_file_paths.join(', ')
        : '';
      // Also populate eval fields so they stay in sync on config load
      this.evalAgentConfigFilePath = s.agent_config_file_path || '';
      this.evalModel = s.model || d.model;
      this.evalProvider = s.provider || d.provider;
    },

    // ── Directory/File Browser ────────────────────
    openBrowser(currentValue, mode = 'directory', fileFilter = null) {
      return new Promise((resolve) => {
        this._browserResolve = resolve;
        this.browserMode = mode;
        this.browserFileFilter = fileFilter;
        const start = currentValue || '~';
        this._browseDir(start);
        this.showBrowser = true;
      });
    },

    async _browseDir(path) {
      const resp = await fetch(`/api/fs/browse?path=${encodeURIComponent(path)}`);
      const data = await resp.json();
      this.browserPath = data.current;
      this.browserParent = data.parent;
      // In file mode, filter files by extension if specified
      if (this.browserMode === 'file' && this.browserFileFilter) {
        this.browserEntries = data.entries.filter(e =>
          e.type === 'directory' || this.browserFileFilter.some(ext => e.name.endsWith(ext))
        );
      } else {
        this.browserEntries = data.entries;
      }
    },

    browserSelect() {
      if (this._browserResolve) this._browserResolve(this.browserPath);
      this.showBrowser = false;
    },

    browserSelectFile(filePath) {
      if (this._browserResolve) this._browserResolve(filePath);
      this.showBrowser = false;
    },

    browserCancel() {
      if (this._browserResolve) this._browserResolve(null);
      this.showBrowser = false;
    },

    browserDblClick(entry) {
      if (entry.type === 'directory') {
        this._browseDir(entry.path);
      } else if (this.browserMode === 'file') {
        this.browserSelectFile(entry.path);
      }
    },

    async browseFor(field) {
      const configFileFields = ['agentConfigFilePath', 'evalAgentConfigFilePath', 'evalSimulationFilePath'];
      if (configFileFields.includes(field)) {
        const val = await this.openBrowser(this[field], 'file', ['.json', '.yaml', '.yml']);
        if (val) this[field] = val;
      } else {
        const val = await this.openBrowser(this[field]);
        if (val) this[field] = val;
      }
    },

    async saveConfig(path) {
      const settings = {
        agent_config_file_path: this.agentConfigFilePath,
        model: this.model,
        provider: this.provider,
        num_conversations: parseInt(this.numConversations) || this._defaults.numConversations,
        max_turns: parseInt(this.maxTurns) || this._defaults.maxTurns,
        num_workers: this.numWorkers || 'auto',
      };
      const resp = await fetch('/api/fs/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ settings, path: path || null }),
      });
      const data = await resp.json();
      if (data.error) {
        alert('Failed to save: ' + data.error);
      } else {
        this.selectedConfig = data.path;
        await this._loadConfigs();
      }
    },

    async saveEvalConfig(path) {
      const settings = {
        agent_config_file_path: this.evalAgentConfigFilePath || this.agentConfigFilePath,
        model: this.evalModel,
        provider: this.evalProvider,
        num_workers: this.evalNumWorkers || 'auto',
        generate_html_report: this.generateHtmlReport,
        score_threshold: this.scoreThreshold ? parseFloat(this.scoreThreshold) : undefined,
      };
      const savePath = path || null;
      const resp = await fetch('/api/fs/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ settings, path: savePath }),
      });
      const data = await resp.json();
      if (data.error) {
        alert('Failed to save: ' + data.error);
      } else {
        await this._loadConfigs();
      }
    },

    async saveConfigAs() {
      const startDir = this.selectedConfig
        ? this.selectedConfig.substring(0, this.selectedConfig.lastIndexOf('/'))
        : '~';
      const val = await this.openBrowser(startDir, 'directory');
      if (!val) return;
      const name = prompt('Config file name:', 'config_simulate.yaml');
      if (!name) return;
      await this.saveConfig(val + '/' + name);
    },

    async saveEvalConfigAs() {
      const startDir = this.selectedConfig
        ? this.selectedConfig.substring(0, this.selectedConfig.lastIndexOf('/'))
        : '~';
      const val = await this.openBrowser(startDir, 'directory');
      if (!val) return;
      const name = prompt('Config file name:', 'config_evaluate.yaml');
      if (!name) return;
      await this.saveEvalConfig(val + '/' + name);
    },

    async browseForConfig() {
      const startDir = this.selectedConfig
        ? this.selectedConfig.substring(0, this.selectedConfig.lastIndexOf('/'))
        : '~';
      const val = await this.openBrowser(startDir, 'file', ['.yaml', '.yml']);
      if (val) this._applyConfig(val);
    },

    // ── Scenario Builder ─────────────────────────
    async _loadDemoScenario() {
      try {
        const resp = await fetch('/api/fs/scenario/demo');
        const data = await resp.json();
        if (data.error || !data.items) return;
        this._scenariosFromJson(data);
        this.scenarioFilePath = data._path || '';
        this.scenarioDirty = false;
        this.scenarioIsDemo = true;
      } catch { /* ignore — demo not available */ }
    },

    _newScenario(idx) {
      return {
        scenario_id: `scenario-${String(idx).padStart(3, '0')}`,
        user_id: `user-${String(idx).padStart(3, '0')}`,
        goal: '',
        _knowledgeText: '',
        user_profile: '',
        origin: { source: 'ui', method: 'manual' },
      };
    },

    addScenario() {
      this.scenarios.push(this._newScenario(this.scenarios.length + 1));
      this.scenarioDirty = true;
    },

    removeScenario(idx) {
      this.scenarios.splice(idx, 1);
      this.scenarioDirty = true;
    },

    _scenariosToJson() {
      return {
        schema_version: '1.0',
        items: this.scenarios.map((sc, i) => {
          const origin = { ...(sc.origin || { source: 'ui', method: 'manual' }) };
          return {
            scenario_id: sc.scenario_id || `scenario-${String(i + 1).padStart(3, '0')}`,
            user_id: sc.user_id || `user-${String(i + 1).padStart(3, '0')}`,
            goal: sc.goal,
            knowledge: (sc._knowledgeText || '').trim()
              ? [{ content: sc._knowledgeText.trim() }]
              : [],
            user_profile: sc.user_profile || '',
            origin,
          };
        }),
      };
    },

    _scenariosFromJson(data) {
      this.scenarios = (data.items || data.scenarios || []).map((item, i) => ({
        scenario_id: item.scenario_id || `scenario-${String(i + 1).padStart(3, '0')}`,
        user_id: item.user_id || `user-${String(i + 1).padStart(3, '0')}`,
        goal: item.goal || '',
        _knowledgeText: (item.knowledge || []).map(k => k.content || '').join('\n'),
        user_profile: item.user_profile || '',
        origin: item.origin || {},
      }));
    },

    async browseForScenario() {
      const startDir = this.scenarioFilePath
        ? this.scenarioFilePath.substring(0, this.scenarioFilePath.lastIndexOf('/'))
        : '~';
      const val = await this.openBrowser(startDir, 'file', ['.json']);
      if (val) {
        this.scenarioFilePath = val;
        await this.loadScenarioFile();
      }
    },

    async loadScenarioFile() {
      if (!this.scenarioFilePath) return;
      try {
        const resp = await fetch(`/api/fs/scenario?path=${encodeURIComponent(this.scenarioFilePath)}`);
        const data = await resp.json();
        if (data.error) { alert(data.error); this.scenarioFileValid = false; return; }
        this._scenariosFromJson(data);
        this.scenarioDirty = false;
        this.scenarioIsDemo = false;
        this.scenarioFileValid = true;
      } catch (e) {
        alert('Failed to load: ' + e.message);
        this.scenarioFileValid = false;
      }
    },

    async validateScenarioFile() {
      if (!this.scenarioFilePath || !this.scenarioFilePath.toLowerCase().endsWith('.json')) {
        this.scenarioFileValid = false;
        return;
      }
      try {
        const resp = await fetch(`/api/fs/scenario?path=${encodeURIComponent(this.scenarioFilePath)}`);
        const data = await resp.json();
        const items = data.items ?? data.scenarios ?? [];
        this.scenarioFileValid = !data.error && Array.isArray(items);
      } catch {
        this.scenarioFileValid = false;
      }
    },

    async saveScenarioFile() {
      if (this.scenarios.length === 0) return;
      let savePath;
      if (this.scenarioFilePath) {
        savePath = this.scenarioFilePath;
      } else {
        const dir = await this.openBrowser('~', 'directory');
        if (!dir) return;
        savePath = dir + '/scenario.json';
      }
      try {
        const resp = await fetch('/api/fs/scenario', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ data: this._scenariosToJson(), path: savePath }),
        });
        const result = await resp.json();
        if (result.error) { alert(result.error); return; }
        this.scenarioFilePath = result.path;
        this.scenarioDirty = false;
      } catch (e) {
        alert('Failed to save: ' + e.message);
      }
    },

    async saveScenarioFileAs() {
      if (this.scenarios.length === 0) return;
      const startDir = this.scenarioFilePath
        ? this.scenarioFilePath.substring(0, this.scenarioFilePath.lastIndexOf('/'))
        : '~';
      const dir = await this.openBrowser(startDir, 'directory');
      if (!dir) return;
      // Simulator expects the file to be named scenario.json
      const savePath = dir + '/scenario.json';
      try {
        const resp = await fetch('/api/fs/scenario', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ data: this._scenariosToJson(), path: savePath }),
        });
        const result = await resp.json();
        if (result.error) { alert(result.error); return; }
        this.scenarioFilePath = result.path;
        this.scenarioDirty = false;
      } catch (e) {
        alert('Failed to save: ' + e.message);
      }
    },

    // ── Result Helpers ──────────────────────────
    _outputFiles: [
      { name: 'final_report.html', icon: 'description', desc: 'Full report with metrics, assessment, and top errors with suggested fixes' },
      { name: 'evaluation.json', icon: 'data_object', desc: 'Full evaluation results: per-turn scores, unique errors, and overall agent performance' },
    ],

    resultFileUrl(name) {
      const dir = this.resultsInputSource === 'auto' ? this.evalOutputDir : this.resultsInputDir;
      if (!dir) return '#';
      return `/api/results/file?dir=${encodeURIComponent(dir)}&name=${encodeURIComponent(name)}`;
    },

    get resultMetrics() {
      const m = this.evalResults?.agent_convo_metrics;
      if (!m || m.length === 0) return null;
      const total = m.length;
      const passCount = m.filter(c => c.status === 'Done').length;
      const avgTurns = m.reduce((s, c) => s + c.turns, 0) / total;
      const overall = m.reduce((s, c) => s + c.final_score, 0) / total;
      return {
        overall: overall.toFixed(2),
        passRate: (passCount / total * 100).toFixed(0) + '%',
        avgTurns: avgTurns.toFixed(1),
        total: String(total),
      };
    },

    get reportUrl() {
      const dir = this.resultsInputSource === 'auto' ? this.evalOutputDir : this.resultsInputDir;
      if (!dir) return null;
      return `/api/results/report?dir=${encodeURIComponent(dir)}`;
    },

    statusBadge(status) {
      const d = this.darkMode;
      const map = {
        idle:    d ? 'bg-zinc-700 text-zinc-400' : 'bg-gray-100 text-gray-600',
        running: d ? 'bg-yellow-900/30 text-yellow-400' : 'bg-yellow-100 text-yellow-700',
        done:    d ? 'bg-green-900/30 text-green-400' : 'bg-green-100 text-green-700',
        failed:    d ? 'bg-red-900/30 text-red-400' : 'bg-red-100 text-red-700',
        cancelled: d ? 'bg-orange-900/30 text-orange-400' : 'bg-orange-100 text-orange-700',
      };
      return map[status] || map.idle;
    },
  };
}
