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

  @OnJob({ name: JobName.AssetDetectAnimalsQueueAll, queue: QueueName.SmartSearch })
  async queueAll({ force }: { force?: boolean } = {}): Promise<JobStatus> {
    // get config to check enabled
    const { machineLearning } = await this.getConfig({ withCache: false });
    if (!machineLearning?.animalRecognition?.enabled) {
      return JobStatus.Skipped;
    }

    let jobs = [];
    const assetsStream = this.assetJobRepository.streamForDetectAnimalsJob(force);
    for await (const asset of assetsStream) {
      jobs.push({ name: JobName.AssetDetectAnimals, data: { id: asset.id } });
      // batch/queue in chunks if needed
    }
    if (jobs.length > 0) {
      await this.jobRepository.queueAll(jobs);
    }
    return JobStatus.Completed;
  }

  @OnJob({ name: JobName.AssetDetectAnimals, queue: QueueName.SmartSearch })
  @Chunked({ size: JOBS_ASSET_PAGINATION_SIZE })
  async processPage(page: number) {
    const assets = await this.assetRepository.getAssetsPageForDetection(page, JOBS_ASSET_PAGINATION_SIZE);
    for (const asset of assets) {
      try {
        if (asset.visibility === 'Hidden') continue;

        const record = await this.assetJobRepository.getForDetectAnimalsJob(asset.id);
        const previewFile = record?.files?.[0];
        if (!previewFile) continue;

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

        const prefix = 'Animals';
        const tagNames = animals.map((a) => `${prefix}/${a.label}`);
        const tags = await upsertTags(this.tagRepository, { userId: asset.ownerId, tags: tagNames });

        for (const t of tags) {
          await this.tagRepository.addAssetIds(t.id, [asset.id]);
        }

        await this.assetRepository.upsertJobStatus({ assetId: asset.id, animalsRecognizedAt: new Date() });
      } catch (err) {
        this.logger.error(`Failed to detect animals for asset ${asset.id}: ${err instanceof Error ? err.message : String(err)}`);
      }
    }
    return JobStatus.Completed;
  }
}
