# Streamlit Cloud Deployment Guide

## Prerequisites

1. **GitHub Repository**: Your code must be in a public GitHub repository
2. **Model File**: Ensure `FV2.h5` is in your repository
3. **Requirements**: All dependencies in `requirements.txt`

## Deployment Steps

### 1. Push to GitHub
```bash
git add .
git commit -m "Ready for Streamlit Cloud deployment"
git push origin main
```

### 2. Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Set main file path: `App.py`
6. Click "Deploy"

## Configuration Files

### `requirements.txt`
- Specifies Python dependencies
- Uses compatible versions for Streamlit Cloud
- NumPy < 2.0.0 to avoid TensorFlow conflicts

### `packages.txt`
- Specifies system dependencies
- Required for TensorFlow on Linux

### `.streamlit/config.toml`
- App configuration
- Optimized for cloud deployment

### `.streamlit/secrets.toml`
- API keys and sensitive data
- Configure in Streamlit Cloud dashboard

## Troubleshooting

### Common Issues

#### 1. NumPy Compatibility Error
**Error**: `numpy.core._multiarray_umath failed to import`
**Solution**: Use `numpy<2.0.0` in requirements.txt

#### 2. Model Loading Error
**Error**: `Model file not found`
**Solution**: Ensure `FV2.h5` is in your repository

#### 3. Memory Issues
**Error**: `Out of memory`
**Solution**: 
- Reduce model size
- Use smaller images
- Optimize imports

#### 4. Import Errors
**Error**: `Module not found`
**Solution**: Check requirements.txt and ensure all dependencies are listed

### Performance Optimization

1. **Model Optimization**:
   - Use TensorFlow Lite for smaller models
   - Quantize model weights
   - Use model compression

2. **Image Processing**:
   - Resize images before processing
   - Use efficient image formats
   - Implement caching

3. **Memory Management**:
   - Clear variables after use
   - Use generators for large datasets
   - Implement garbage collection

## Environment Variables

Set these in Streamlit Cloud dashboard:

```toml
GEMINI_API_KEY = "your-gemini-api-key"
```

## Monitoring

- Check app logs in Streamlit Cloud dashboard
- Monitor memory usage
- Track API call limits
- Monitor error rates

## Best Practices

1. **Keep Dependencies Minimal**: Only include necessary packages
2. **Use Specific Versions**: Avoid version conflicts
3. **Handle Errors Gracefully**: Provide user-friendly error messages
4. **Optimize for Cloud**: Consider resource limitations
5. **Test Locally**: Ensure app works before deploying

## Support

- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-community-cloud)
- [TensorFlow Cloud Deployment](https://www.tensorflow.org/guide/keras/save_and_serialize)
- [GitHub Issues](https://github.com/streamlit/streamlit/issues) 