import { Kysely, sql } from 'kysely';

export async function up(db: Kysely<any>): Promise<void> {
  await sql`ALTER TABLE "asset_job_status" ADD COLUMN IF NOT EXISTS "animalsRecognizedAt" timestamptz NULL;`.execute(db);
}

export async function down(db: Kysely<any>): Promise<void> {
  await sql`ALTER TABLE "asset_job_status" DROP COLUMN IF EXISTS "animalsRecognizedAt";`.execute(db);
}
