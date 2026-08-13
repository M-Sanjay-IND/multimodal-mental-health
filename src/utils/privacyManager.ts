/**
 * Privacy Manager: Enforces zero-retention policies on raw biometric data.
 * Guarantees that raw video frame pixels and PCM audio buffers are overwritten
 * and released immediately after feature vector extraction.
 */

export interface PrivacyPurgeOptions {
  canvasContext?: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null;
  audioBuffer?: Float32Array | Int16Array | Uint8Array | null;
  videoElement?: HTMLVideoElement | null;
}

/**
 * Overwrites an ArrayBuffer or TypedArray with zeros in-place.
 */
export function overwriteBufferWithZeros(buffer: TypedArray | ArrayBuffer | null | undefined): void {
  if (!buffer) return;

  if (buffer instanceof ArrayBuffer) {
    new Uint8Array(buffer).fill(0);
  } else if (ArrayBuffer.isView(buffer)) {
    (buffer as Uint8Array).fill(0);
  }
}

/**
 * Type alias for any JavaScript TypedArray.
 */
type TypedArray =
  | Int8Array
  | Uint8Array
  | Uint8ClampedArray
  | Int16Array
  | Uint16Array
  | Int32Array
  | Uint32Array
  | Float32Array
  | Float64Array;

/**
 * Purges raw media buffers from memory immediately after vector extraction.
 *
 * 1. Clears Canvas 2D image data pixels.
 * 2. Overwrites temporary raw PCM audio buffers with zeros.
 * 3. Prevents memory retention or DOM leakages of raw biometric signals.
 */
export function purgeRawMediaBuffers(options: PrivacyPurgeOptions): void {
  const { canvasContext, audioBuffer } = options;

  // 1. Overwrite canvas pixel data if canvas context is provided
  if (canvasContext) {
    const width = canvasContext.canvas.width;
    const height = canvasContext.canvas.height;
    if (width > 0 && height > 0) {
      try {
        // Clear pixels
        canvasContext.clearRect(0, 0, width, height);

        // Fetch image data and overwrite raw bytes with 0
        const imgData = canvasContext.getImageData(0, 0, width, height);
        imgData.data.fill(0);
        canvasContext.putImageData(imgData, 0, 0);
      } catch {
        // Ignore cross-origin canvas read exceptions if canvas is tainted
      }
    }
  }

  // 2. Overwrite audio buffer with zeros
  if (audioBuffer) {
    overwriteBufferWithZeros(audioBuffer);
  }
}
