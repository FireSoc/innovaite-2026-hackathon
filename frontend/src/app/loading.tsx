/* eslint-disable @next/next/no-img-element */
export default function Loading() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-surface-dim">
      <div className="flex flex-col items-center gap-4">
        <img src="/logo.png" alt="Remedy" width={40} height={40} className="h-10 w-10 object-contain animate-pulse" />
        <p className="text-sm text-text-muted">Loading Remedy...</p>
      </div>
    </div>
  );
}
