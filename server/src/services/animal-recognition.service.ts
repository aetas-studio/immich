import { Injectable, Logger } from '@nestjs/common';
import { OnJob, Chunked } from 'src/decorators';
import { JOBS_ASSET_PAGINATION_SIZE } from 'src/constants';
import { JobName, JobStatus, QueueName, ImmichWorker } from 'src/enum';
import { MachineLearningRepository } from 'src/repositories/machine-learning.repository';
import { TagRepository } from 'src/repositories/tag.repository';
import { AssetRepository } from 'src/repositories/asset.repository';
import { upsertTags } from 'src/utils/tag';

@Injectable()
export class AnimalRecognitionService {
  private readonly logger = new Logger(AnimalRecognitionService.name);

  constructor(
    private readonly machineLearningRepository: MachineLearningRepository,
    private readonly tagRepository: TagRepository,
    private readonly assetRepository: AssetRepository,
  ) {}

  /**
   * Queue all assets job (same pattern as faces)
   */
  @OnJob({ name: JobName.AssetDetectAnimalsQueueAll, queue: QueueName.SmartSearch })
  async queueAll() {
    await this.assetRepository.queueAllForAnimalDetection();
  }

  /**
   * Process a page of assets to detect animals.
   * This is intentionally similar to the face detection job in person.service.ts
   */
  @OnJob({ name: JobName.AssetDetectAnimals, queue: QueueName.SmartSearch })
  @Chunked({ size: JOBS_ASSET_PAGINATION_SIZE })
  async processPage(page: number) {
    // fetch assets page (use similar repo method as face detection)
    const assets = await this.assetRepository.getAssetsPageForDetection(page, JOBS_ASSET_PAGINATION_SIZE);

    for (const asset of assets) {
      try {
        // skip if hidden,video, etc
        if (asset.visibility === 'Hidden') continue;

        // get preview path for that asset (the repository has methods to get preview file path)
        const previewFile = await this.assetRepository.getPreviewFile(asset.id);
        if (!previewFile) continue;

        const mlConfig = {
          modelName: null,
          minScore: 0.35,
        };

        const { animals } = await this.machineLearningRepository.detectAnimals(
          // read ML URLs from config; machineLearningRepository consumer code usually passes config.urls from server config
          // the calling code that schedules this job should supply the configured URLs; see person.service for example usage
          // here we rely on consumer to inject urls; alternatively, pass server config via DI.
          // For simplicity, using process.env.MACHINE_LEARNING_URLS split or reading repo default
          process.env.MACHINE_LEARNING_URLS ? process.env.MACHINE_LEARNING_URLS.split(',') : [],
          previewFile.path,
          mlConfig,
        );

        if (!animals || animals.length === 0) {
          continue;
        }

        const prefix = 'Animals'; // adjust if you prefer 'Animal' or localized name (check)
        const tagNames = animals.map(a => `${prefix}/${a.label}`);

        const tags = await upsertTags(this.tagRepository, { userId: asset.ownerId, tags: tagNames });

        const tagIdMap = new Map(tags.map(t => [t.value, t.id]));

        for (const t of tags) {
          await this.tagRepository.addAssetIds(t.id, [asset.id]);
        }
      } catch (err) {
        this.logger.error(`Failed to detect animals for asset ${asset.id}: ${err instanceof Error ? err.message : String(err)}`);
      }
    }

    return JobStatus.Completed;
  }
}
