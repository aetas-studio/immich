import { Injectable } from '@nestjs/common';
import { Logger } from '@nestjs/common';
import { MachineLearningRepository } from 'src/repositories/machine-learning.repository';
import { TagRepository } from 'src/repositories/tag.repository';
import { AssetRepository } from 'src/repositories/asset.repository';
import { AssetJobRepository } from 'src/repositories/asset-job.repository';
import { JobRepository } from 'src/repositories/job.repository';
import { SystemConfigService } from 'src/services/system-config.service';
import { upsertTags } from 'src/utils/tag';
import { JobName, JobStatus, QueueName } from 'src/enum';
import { OnJob } from 'src/decorators';

@Injectable()
export class AnimalRecognitionService {
  private readonly logger = new Logger(AnimalRecognitionService.name);

  constructor(
    private readonly mlRepo: MachineLearningRepository,
    private readonly tagRepository: TagRepository,
    private readonly assetRepository: AssetRepository,
    private readonly assetJobRepository: AssetJobRepository,
    private readonly jobRepository: JobRepository,
    private readonly systemConfigService: SystemConfigService,
  ) {}

  // queueAll: mette in coda jobs per singolo asset
  @OnJob({ name: JobName.AssetDetectAnimalsQueueAll, queue: QueueName.SmartSearch })
  async queueAll({ force }: { force?: boolean } = {}) {
    const cfg = await this.systemConfigService.getConfig({ withCache: false });
    if (!cfg.machineLearning?.animalRecognition?.enabled) return JobStatus.Skipped;

    const stream = this.assetJobRepository.streamForDetectAnimalsJob(force);
    const batch: { name: JobName; data: { id: string } }[] = [];
    for await (const asset of stream) {
      batch.push({ name: JobName.AssetDetectAnimals, data: { id: asset.id } as any });
      if (batch.length >= 1000) {
        await this.jobRepository.queueAll(batch as any);
        batch.length = 0;
      }
    }
    if (batch.length) await this.jobRepository.queueAll(batch as any);
    return JobStatus.Success; // usa il membro corretto come spiegato sotto
  }

  // process single job, payload { id }
  @OnJob({ name: JobName.AssetDetectAnimals, queue: QueueName.SmartSearch })
  async processSingle(data: { id: string }) {
    const assetId = data.id;
    const record = await this.assetJobRepository.getForDetectAnimalsJob(assetId);
    if (!record) return JobStatus.Success;

    const preview = record.files?.[0];
    if (!preview) {
      await this.assetRepository.upsertJobStatus({ assetId, animalsRecognizedAt: new Date() });
      return JobStatus.Success;
    }

    const cfg = await this.systemConfigService.getConfig({ withCache: false });
    const mlCfg = {
      modelName: cfg.machineLearning?.animalRecognition?.modelName ?? null,
      minScore: cfg.machineLearning?.animalRecognition?.minScore ?? 0.5,
    };

    const { animals } = await this.mlRepo.detectAnimals(cfg.machineLearning.urls, preview.path, mlCfg);

    if (animals && animals.length > 0) {
      const prefix = 'Animals';
      const tagNames = animals.map((a) => `${prefix}/${a.label}`);
      const tags = await upsertTags(this.tagRepository, { userId: record.ownerId, tags: tagNames });
      for (const t of tags) {
        await this.tagRepository.addAssetIds(t.id, [assetId]);
      }
    }

    await this.assetRepository.upsertJobStatus({ assetId, animalsRecognizedAt: new Date() });
    return JobStatus.Success;
  }
}
