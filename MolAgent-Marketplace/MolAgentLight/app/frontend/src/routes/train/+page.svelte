<script lang="ts">
	import { pipeline, resetPipeline } from '$lib/stores/pipeline.svelte';
	import { datasets, refreshDatasets } from '$lib/stores/datasets.svelte';
	import FileUpload from '$lib/components/shared/FileUpload.svelte';
	import DetectPanel from '$lib/components/train/DetectPanel.svelte';
	import ConfigPanel from '$lib/components/train/ConfigPanel.svelte';
	import PipelineSteps from '$lib/components/train/PipelineSteps.svelte';
	import type { UploadResult, DatasetEntry } from '$lib/api/client';

	$effect(() => { if (pipeline.step === 'upload') refreshDatasets(); });

	function onFileUploaded(result: UploadResult) {
		pipeline.datasetId = result.dataset_id || null;
		pipeline.csvPath = result.path;
		pipeline.csvFilename = result.filename;
		pipeline.step = 'detect';
	}

	function selectDataset(ds: DatasetEntry) {
		pipeline.datasetId = ds.id;
		pipeline.csvPath = null;
		pipeline.csvFilename = ds.filename;
		pipeline.step = 'detect';
	}
</script>
<div class="tp">
	<div class="ph"><h2>Train Model</h2>{#if pipeline.step !== 'upload'}<button class="btn-reset" onclick={resetPipeline}>New Pipeline</button>{/if}</div>
	{#if pipeline.step === 'upload'}
		<FileUpload label="Upload training dataset (CSV)" onUploaded={onFileUploaded} />
		{#if datasets.entries.length > 0}
			<div class="existing-datasets">
				<h3>Or select an existing dataset</h3>
				<div class="ds-list">
					{#each datasets.entries as ds}
						<button class="ds-card" onclick={() => selectDataset(ds)}>
							<span class="ds-name">{ds.filename}</span>
							<span class="ds-meta">{ds.row_count.toLocaleString()} rows &middot; {ds.columns.length} cols</span>
						</button>
					{/each}
				</div>
			</div>
		{/if}
	{/if}
	{#if pipeline.step === 'detect' || pipeline.step === 'configure' || pipeline.step === 'training' || pipeline.step === 'done'}
		{#if pipeline.csvFilename}<p class="fi">Dataset: <span class="mono">{pipeline.csvFilename}</span></p>{/if}
		<DetectPanel />
	{/if}
	{#if pipeline.step === 'configure'}<ConfigPanel />{/if}
	{#if pipeline.step === 'training' || pipeline.step === 'done'}<PipelineSteps />{/if}
	{#if pipeline.error}<p class="err">{pipeline.error}</p>{/if}
</div>
<style>
	.tp { max-width: 900px; display: flex; flex-direction: column; gap: 20px; }
	.ph { display: flex; align-items: center; justify-content: space-between; }
	h2 { font-size: 20px; font-weight: 700; color: var(--text-primary); }
	.btn-reset { padding: 6px 14px; border: 1px solid var(--border); border-radius: 6px; background: none; color: var(--text-secondary); font-size: 12px; cursor: pointer; }
	.btn-reset:hover { background: var(--bg-tertiary); }
	.fi { font-size: 13px; color: var(--text-secondary); }
	.err { font-size: 13px; color: var(--error); padding: 12px; border-radius: var(--radius); background: rgba(239, 68, 68, 0.1); }
	.existing-datasets { border-top: 1px solid var(--border); padding-top: 16px; }
	.existing-datasets h3 { font-size: 13px; font-weight: 500; color: var(--text-muted); margin-bottom: 10px; }
	.ds-list { display: flex; flex-wrap: wrap; gap: 8px; }
	.ds-card { display: flex; flex-direction: column; gap: 2px; padding: 10px 14px; border: 1px solid var(--border); border-radius: var(--radius); background: var(--bg-secondary); cursor: pointer; text-align: left; transition: border-color 120ms, background 120ms; }
	.ds-card:hover { border-color: var(--accent); background: var(--accent-dim); }
	.ds-name { font-size: 12px; font-weight: 500; color: var(--text-primary); font-family: var(--font-mono); }
	.ds-meta { font-size: 11px; color: var(--text-muted); }
</style>
