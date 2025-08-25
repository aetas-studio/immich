import { Injectable, Logger } from '@nestjs/common';
import { BaseService } from 'src/services/base.service';
import { OnJob, Chunked } from 'src/decorators';
import { JOBS_ASSET_PAGINATION_SIZE } from 'src/constants';
import { JobName, JobStatus, QueueName } from 'src/enum';
import { MachineLearningRepository } from 'src/repositories/machine-learning.repository';
import { TagRepository } from 'src/repositories/tag.repository';
import { AssetRepository } from 'src/repositories/asset.repository';
import { upsertTags } from 'src/utils/tag';

@Injectable()
export class AnimalRecognitionService extends BaseService {
  private readonly logger = new Logger(AnimalRecognitionService.name);

  constructor(
    private readonly mlRepo: MachineLearningRepository,
    private readonly tagRepository: TagRepository,
    private readonly assetRepository: AssetRepository,
  ) {
    super();
  }

  /**
   * Queue all assets for animal detection (similar to faces QueueAll handler).
   * This will stream asset ids via assetJobRepository.streamForDetectAnimalsJob
   * and create jobs AssetDetectAnimals for each asset.
   */
  @OnJob({ name: JobName.AssetDetectAnimalsQueueAll, queue: QueueName.SmartSearch })
  async queueAll({ force }: { force?: boolean } = {}): Promise<JobStatus> {
    const { machineLearning } = await this.getConfig({ withCache: false });
    if (!machineLearning?.animalRecognition?.enabled) {
      return JobStatus.Skipped;
    }

    // If force is true we might want to reset previous job timestamps etc. (follow face logic if needed).
    const jobs: { name: JobName; data: { id: string } }[] = [];
    const assetsStream = this.assetJobRepository.streamForDetectAnimalsJob(force);
    for await (const asset of assetsStream) {
      jobs.push({ name: JobName.AssetDetectAnimals, data: { id: asset.id } });
      if (jobs.length >= 1000) {
        await this.jobRepository.queueAll(jobs);
        jobs.length = 0;
      }
    }
    if (jobs.length > 0) {
      await this.jobRepository.queueAll(jobs);
    }
    return JobStatus.Completed;
  }

  /**
   * Process a page of assets (chunked by JOBS_ASSET_PAGINATION_SIZE)
   * This handler will be executed by the worker: for each asset it will call the ML repo,
   * upsert tags (Animals/<Label>) and attach tags to the asset.
   */
  @OnJob({ name: JobName.AssetDetectAnimals, queue: QueueName.SmartSearch })
  @Chunked({ size: JOBS_ASSET_PAGINATION_SIZE })
  async processPage(page: number): Promise<JobStatus> {
    // Fetch a page of assets to process (reuse existing repository helper if present).
    // We use assetsWithPreviews stream pattern in queueAll; here we can fetch a page via an existing helper if available.
    const assets = await this.assetRepository.getAssetsPageForDetection?.(page, JOBS_ASSET_PAGINATION_SIZE) ?? [];

    for (const asset of assets) {
      try {
        if (asset.visibility === 'Hidden') continue;

        const record = await this.assetJobRepository.getForDetectAnimalsJob(asset.id);
        const previewFile = record?.files?.[0];
        if (!previewFile) {
          // mark processed to avoid retry depending on your policy
          await this.assetRepository.upsertJobStatus({ assetId: asset.id, animalsRecognizedAt: new Date() });
          continue;
        }

        const { machineLearning } = await this.getConfig({ withCache: false });
        const mlConfig = {
          modelName: machineLearning?.animalRecognition?.modelName ?? null,
          minScore: machineLearning?.animalRecognition?.minScore ?? 0.5,
        };

        const { animals } = await this.mlRepo.detectAnimals(
          machineLearning.urls,
          previewFile.path,
          mlConfig,
        );

        if (!animals || animals.length === 0) {
          await this.assetRepository.upsertJobStatus({ assetId: asset.id, animalsRecognizedAt: new Date() });
          continue;
        }

        // Build tags like 'Animals/Dog', 'Animals/Cat'
        const prefix = 'Animals';
        const tagNames = animals.map((a) => `${prefix}/${a.label}`);

        // Upsert tags and attach them to the asset
        const tags = await upsertTags(this.tagRepository, { userId: asset.ownerId, tags: tagNames });

        for (const t of tags) {
          // addAssetIds will ignore duplicates and add mapping
          await this.tagRepository.addAssetIds(t.id, [asset.id]);
        }

        await this.assetRepository.upsertJobStatus({ assetId: asset.id, animalsRecognizedAt: new Date() });
      } catch (err) {
        // fixed logging, no typo
        this.logger.error(`Failed to detect animals for asset ${asset.id}: ${err instanceof Error ? err.message : String(err)}`);
      }
    }

    return JobStatus.Completed;
  }
}
